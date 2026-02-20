import logging
import os
import time
import re
import json
import dotenv
import prompt
import torch
import wandb
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from openai import OpenAI

# Initialize logging for better visibility of the training process
logging.basicConfig(level=logging.INFO)

# Disable online syncing for Weights & Biases by default to avoid accidental data upload
os.environ["WANDB_MODE"] = "offline"

# Load environment configurations from backend/.env (API keys, model paths, etc.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
dotenv.load_dotenv(os.path.join(BACKEND_DIR, '.env'))

# DeepSeek API Configuration (Used for LLM-as-a-Judge reward calculation)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
USE_DEEPSEEK_JUDGE = os.getenv("USE_DEEPSEEK_JUDGE", "true").lower() == "true"

# ---------- Hyperparameters & Configuration ----------
NAME = os.getenv("OUTLINE_NAME", "outline01")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("OUTLINE_PROJECT", "outline-training")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 4096))

print(f"Project: {PROJECT_NAME} | Run: {NAME} | Base Model: {MODEL_NAME}")
print(f"Max Sequence Length for Training: {MAX_SEQ_LEN}")

# W&B Experiment Tracking Setup
WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}")

# Model Cache Directory (Crucial for large clusters with limited home space)
MODEL_CACHE_DIR = "/data/xxx/model" 

# Global placeholders for the model and tokenizer to prevent redundant reloading
tokenizer = None
model = None

def load_model_and_tokenizer():
    """
    Loads the pre-trained model and tokenizer into VRAM.
    Uses bfloat16 for a balance between precision and memory efficiency.
    """
    global tokenizer, model
    
    if tokenizer is None:
        print(f"[Model] Initializing tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    if model is None:
        print(f"[Model] Loading causal LM: {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.bfloat16
        )
        
        # Display hardware distribution for multi-GPU setups
        if hasattr(model, 'hf_device_map'):
            print(f"[GPU] Device Map distribution: {model.hf_device_map}")
        else:
            print(f"[GPU] Running on device: {next(model.parameters()).device}")
    
    return model, tokenizer


# ============================================================
# REWARD FUNCTION 1: Rule-Based (Structure, Syntax & Format)
# ============================================================
def rule_based_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Evaluates the quality of a Markdown outline using deterministic rules.
    Normalized score range: [0.0, 1.0].
    
    Logic:
    - Checks for specific Markdown headers (H1, H2, H3).
    - Validates bullet point syntax (starts with verbs, length constraints).
    - Penalizes forbidden sections like 'Conclusion' or 'References' for outlines.
    """
    rewards = []
    for md in completions:
        try:
            score = 0.0
            
            # 1. H1 Header Validation (0.1 pts): Expects exactly one main title
            h1 = re.findall(r"(?m)^# [^\n]+$", md)
            if len(h1) == 1:
                score += 0.1
            
            # 2. H2 Header Validation (0.2 pts): Target is 5 major sections
            h2_matches = list(re.finditer(r"(?m)^## [^\n]+$", md))
            if len(h2_matches) == 5:
                score += 0.2
            elif len(h2_matches) > 0:
                score += 0.1 * min(len(h2_matches), 5) / 5
            
            # 3. H3 Header Validation (0.2 pts): Target 15-20 subsections
            h3_count = len(re.findall(r"(?m)^### [^\n]+$", md))
            if 15 <= h3_count <= 20:
                score += 0.2
            elif h3_count > 0:
                score += 0.1 * min(h3_count, 15) / 15
            
            # 4. Bullet Point Quality (0.3 pts)
            bullets = re.findall(r"(?m)^- (.+)$", md)
            if len(bullets) >= 45:
                # Professional Verb Prefixes (English equivalents for international repos)
                # NOTE: Replace with Chinese verbs if training for Chinese PPT generation.
                verb_prefixes = (
                    "Analyze", "Design", "Implement", "Optimize", "Evaluate", "Build", 
                    "Formulate", "Apply", "Establish", "Develop", "Deploy", "Test", 
                    "Verify", "Plan", "Integrate", "Configure", "Monitor", "Manage", 
                    "Enhance", "Improve", "Create", "Define", "Execute", "Advance", 
                    "Ensure", "Support", "Provide", "Resolve"
                )
                
                verb_count = sum(1 for b in bullets if any(b.strip().lower().startswith(v.lower()) for v in verb_prefixes))
                verb_ratio = verb_count / len(bullets) if bullets else 0
                
                # Length & Punctuation check: Slides should be concise and not end with periods.
                valid_format = sum(1 for b in bullets if len(b.strip()) <= 60 and not b.strip().endswith((".", "!")))
                format_ratio = valid_format / len(bullets) if bullets else 0
                
                score += 0.15 * verb_ratio + 0.15 * format_ratio
            elif len(bullets) > 0:
                score += 0.1 * len(bullets) / 45
            
            # 5. Negative Constraints (0.2 pts base - Penalties)
            penalty = 0.0
            # Header shouldn't be a question
            if re.search(r"^#+ .*[？?]", md, re.MULTILINE):
                penalty += 0.1
            # Forbidden 'Filler' sections (Outlines should be pure content)
            forbidden_pattern = r"(Introduction|Conclusion|Summary|Table of Contents|References|Notes|Disclaimer)"
            if re.search(forbidden_pattern, md, re.IGNORECASE):
                penalty += 0.1
            
            score = max(0.0, score + 0.2 - penalty)
            rewards.append(min(1.0, score))
            
        except Exception:
            rewards.append(0.0)
    
    return rewards


# ============================================================
# REWARD FUNCTION 2: LLM-as-a-Judge (DeepSeek API)
# ============================================================
JUDGE_PROMPT = """Evaluate the quality of the following presentation outline. Provide a score from 0 to 10.

Criteria:
1. Content Quality (0-4): Logical progression, professional depth, practical insights.
2. Structure (0-3): Clear hierarchy, comprehensive topic coverage.
3. Syntax (0-3): Concise points, starts with active verbs, no redundancy.

Outline:
{outline}

Response format: Output ONLY the numerical score (0-10)."""

def _call_deepseek_judge(outline: str) -> float:
    """Invokes DeepSeek API to act as a quality judge; returns normalized score [0, 1]."""
    try:
        base_url = DEEPSEEK_BASE_URL.replace("/chat/completions", "")
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=base_url)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(outline=outline)}],
            temperature=0.1,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        match = re.search(r"(\d+(?:\.\d+)?)", score_text)
        if match:
            score = float(match.group(1))
            return min(1.0, max(0.0, score / 10.0))
        return 0.5
    except Exception as e:
        logging.warning(f"Judge API call failed: {e}")
        return 0.5

def deepseek_judge_reward(completions: List[str], **kwargs) -> List[float]:
    """Semantic reward function using a powerful external LLM for quality assessment."""
    if not USE_DEEPSEEK_JUDGE or not DEEPSEEK_API_KEY:
        return [0.5] * len(completions)
    
    return [_call_deepseek_judge(c) for c in completions]


# ============================================================
# TRAINING DATASET PIPELINE
# ============================================================
def prepare_gspo_dataset(topics: List[str]) -> Dataset:
    """Prepares the dataset for GSPO training by formatting topics into ChatML prompts."""
    prompts = []
    
    for topic in topics:
        messages = [
            {"role": "system", "content": prompt.ROLLOUT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}
        ]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"System: {prompt.ROLLOUT_SYSTEM_PROMPT}\n\nUser: {prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}\n\nAssistant:"
        
        prompts.append(formatted_prompt)
    
    return Dataset.from_dict({"prompt": prompts})


# ============================================================
# MAIN TRAINING EXECUTION
# ============================================================
def main():
    global model, tokenizer
    
    print("[Main] Initializing models and environment...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Initialize Weights & Biases for real-time loss/reward tracking
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY if WANDB_ENTITY else None,
        name=WANDB_RUN_NAME,
        config={
            "run_name": NAME,
            "base_model": MODEL_NAME,
            "algorithm": "GSPO (Group Sequence Policy Optimization)",
        },
        settings=wandb.Settings(start_method="thread"),
    )

    # Load training topics
    script_dir = os.path.dirname(os.path.abspath(__file__))
    topic_json_path = os.path.join(script_dir, 'topic.json')
    assert os.path.exists(topic_json_path), f"Training data not found at: {topic_json_path}"
    
    with open(topic_json_path, 'r', encoding='utf-8') as f:
        topic_data = json.load(f)
    topics = topic_data["topics"]
    print(f"[Data] Successfully loaded {len(topics)} topics.")

    train_dataset = prepare_gspo_dataset(topics)

    # Set up timestamped output directory
    output_dir = f"./output/{NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # GSPO Configuration
    # Note: GSPO optimizes policy via sequence-level importance sampling for better stability.
    gspo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=50,
        max_completion_length=MAX_SEQ_LEN // 2,
        num_generations=4,         # Generates 4 candidates per prompt for variance reduction
        temperature=0.7,
        seed=42,
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,
        beta=0.04,                 # KL Divergence weight
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
    )

    # Define Reward Pipeline
    reward_funcs = [rule_based_reward]
    if USE_DEEPSEEK_JUDGE and DEEPSEEK_API_KEY:
        reward_funcs.append(deepseek_judge_reward)
        print("[Reward] Hybrid Mode: Rule-based + DeepSeek Judge enabled.")
    else:
        print("[Reward] Manual Mode: Only rule-based rewards will be used.")
    
    # Initialize the Trainer
    trainer = GRPOTrainer(
        model=model,
        args=gspo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )

    print(f"[Train] Starting GSPO optimization. Log directory: {output_dir}")
    trainer.train()
    
    # Save final artifacts
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"[Train] Training completed. Fine-tuned model saved at: {output_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()