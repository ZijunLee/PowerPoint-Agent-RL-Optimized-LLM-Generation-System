import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from typing import List, Dict, Any
import re
import json
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
import wandb
from openai import OpenAI
from prompt import CONTENT_SYSTEM_PROMPT, CONTENT_USER_PROMPT, JUDGE_PROMPT

os.environ["WANDB_MODE"] = "offline"

# Configure standard logging
logging.basicConfig(level=logging.INFO)

# Global configuration via Environment Variables
# Pathing ensures .env is loaded from the parent 'backend' directory
dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
USE_DEEPSEEK_JUDGE = os.getenv("USE_DEEPSEEK_JUDGE", "true").lower() == "true"

MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 4096))
MODEL_CACHE_DIR = "/data/xxx/model" 

# ============================================================
# REWARD FUNCTION: Content Quality & Structural Logic
# ============================================================
def content_quality_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Analyzes the generated content and assigns a score based on professionalism.
    Score range: [0.0, 1.0].
    """
    rewards = []
    for content in completions:
        try:
            score = 0.0
            text = content.strip()
            
            # 1. Structural Checklist (0.25 pts): Verification of Markdown hierarchy
            if re.search(r'^# ', text, re.MULTILINE): score += 0.05
            if re.search(r'^## ', text, re.MULTILINE): score += 0.08
            if re.search(r'^### ', text, re.MULTILINE): score += 0.07
            if re.search(r'^- ', text, re.MULTILINE): score += 0.05
            
            # 2. Semantic Richness (0.25 pts): Based on content length
            length = len(text)
            if length >= 2000: score += 0.25
            elif length >= 1000: score += 0.15
            
            # 3. Evidence-Based Logic (0.2 pts): Detection of metrics and data points
            # FULLY ENGLISH VERSION: Removed Asian character markers (亿, 年, 万, etc.)
            data_markers = re.findall(r'\d+[%]|\d+\.\d+|\d{4}(?:year|yr)|\d+(?:billion|bn|million|m)', text, re.IGNORECASE)
            score += 0.2 if len(data_markers) >= 5 else (0.1 if len(data_markers) >= 1 else 0)
            
            # 4. Template Avoidance (0.15 pts): Penalizes generic AI filler text
            bad_patterns = [
                r'^Detailed content', r'^This section introduces', 
                r'^Based on the above', r'^The following is', 
                r'^Here is the content', r'^Please refer to'
            ]
            if not any(re.search(p, text, re.IGNORECASE) for p in bad_patterns):
                score += 0.15
            
            # 5. Professional Terminology (0.15 pts): Verification of domain expertise
            tech_lexicon = [
                'Architecture', 'Optimization', 'Infrastructure', 'Framework', 'Deployment',
                'Methodology', 'Innovation', 'Scalability', 'Integration', 'Analysis',
                'Strategy', 'Implementation', 'Metrics', 'Benchmark', 'Paradigm'
            ]
            matches = sum(1 for term in tech_lexicon if term.lower() in text.lower())
            score += 0.15 if matches >= 10 else (0.1 if matches >= 5 else 0)
            
            rewards.append(min(1.0, score))
        except Exception:
            rewards.append(0.0)
    return rewards

# ============================================================
# PIPELINE: Data Loading & Prompt Templating
# ============================================================
def load_and_template_data(tokenizer):
    """Loads outlines and applies the chat template in a single efficient pass."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../outline/outline.jsonl')
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    prompts = []
    for item in data:
        # Clean markdown artifacts from previous stage
        clean_outline = re.sub(r'^```markdown\s*|\s*```[\s\S]*$', '', item.get("outline", "")).strip()
        
        msg = [
            {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
            {"role": "user", "content": CONTENT_USER_PROMPT.format(topic=item["topic"], outline=clean_outline)}
        ]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    return Dataset.from_dict({"prompt": prompts})

def main():
    # Model Initialization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, device_map="auto", 
        cache_dir=MODEL_CACHE_DIR, torch_dtype=torch.bfloat16
    )

    # Dataset & Config
    dataset = load_and_template_data(tokenizer)
    gspo_args = GRPOConfig(
        output_dir=f"./output/content-{time.strftime('%Y%m%d-%H%M%S')}",
        learning_rate=1e-6, num_train_epochs=1, bf16=True,
        max_completion_length=MAX_SEQ_LEN, num_generations=4,
        importance_sampling_level="sequence", report_to="none"
    )

    # Launch Reinforcement Learning via GRPOTrainer
    trainer = GRPOTrainer(
        model=model, args=gspo_args, train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[content_quality_reward]
    )

    print("[Phase] Starting GSPO Reinforcement Learning for Content Generation...")
    trainer.train()
    trainer.save_model(gspo_args.output_dir)

if __name__ == "__main__":
    main()