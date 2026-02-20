import os
import json
import time
import dotenv
import prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables from .env file
dotenv.load_dotenv()

# ---------- Configuration ----------
# Project and model settings from environment variables
NAME = os.getenv("ART_NAME", "web-search-outline")
BASE_MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# Model cache directory for high-performance clusters or shared storage
MODEL_CACHE_DIR = "/data/xxx/model" 

# Specify the path to your fine-tuned model checkpoint
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(SCRIPT_DIR, "output/outline01-20251220-223525")

# Test topics across various domains to verify generalization
TEST_TOPICS = [
    "Corporate Digital Transformation Strategic Planning",
    "Innovative Applications of LLMs in Education",
    "Smart Manufacturing and Industry 4.0 Transition Paths",
    "FinTech Innovation and Risk Management",
    "Metaverse Technology Development and Business Applications",
    "Green Energy Development under Carbon Neutrality Goals",
    "Blockchain Technology Applications in Supply Chain",
    "Team Management in Remote Work Models",
    "Short Video Marketing Strategies and Content Creation",
    "Cloud-Native Architecture Design and Practice",
    "Commercialization Process of Autonomous Driving Technology",
    "Virtual Digital Human Technology and Application Scenarios",
    "Agricultural Tech Innovation and Smart Agriculture",
]


def load_model():
    """
    Loads the fine-tuned checkpoint from the local filesystem.
    Configures device mapping and precision (bfloat16).
    """
    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"[Model] Loading fine-tuned checkpoint: {TRAINED_MODEL_PATH}")
        model_path = TRAINED_MODEL_PATH
    else:
        # Fallback error if the training output directory is not found
        raise FileNotFoundError(f"Trained model path not found: {TRAINED_MODEL_PATH}")
    
    # Initialize the tokenizer from the checkpoint directory
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        cache_dir=MODEL_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the Causal Language Model with automatic multi-GPU distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16
    )
    
    # Log the device map for hardware monitoring
    if hasattr(model, 'hf_device_map'):
        print(f"[GPU] Device Map Distribution: {model.hf_device_map}")
    else:
        print(f"[GPU] Model loaded on device: {next(model.parameters()).device}")
    
    return model, tokenizer


def generate_outline(model, tokenizer, topic: str) -> str:
    """
    Constructs the prompt and performs text generation for a given topic.
    """
    # Build ChatML message structure
    messages = [
        {"role": "system", "content": prompt.ROLLOUT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}
    ]
    
    # Apply model-specific chat template for optimal formatting
    if hasattr(tokenizer, 'apply_chat_template'):
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback to manual prompt engineering if template is missing
        input_text = f"System: {prompt.ROLLOUT_SYSTEM_PROMPT}\n\nUser: {prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}\n\nAssistant:"
    
    # Move tensors to the appropriate model device
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Perform generation with controlled sampling
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,      # Large limit to allow detailed outlines
            temperature=0.7,          # Balanced creativity and coherence
            do_sample=True,
            top_p=0.9,                # Nucleus sampling for diversity
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Slice outputs to exclude the input prompt tokens
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    outline = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return outline


def run_test(model, tokenizer, topic: str, index: int):
    """
    Executes a single test case and logs the output preview.
    """
    print(f"\n{'='*70}")
    print(f"Test Case {index}/{len(TEST_TOPICS)}: {topic}")
    print('='*70)
    
    outline = generate_outline(model, tokenizer, topic)
    
    print("\nGenerated Outline Preview:")
    print("-" * 70)
    # Print first 500 characters as a preview
    print(outline[:500] + "..." if len(outline) > 500 else outline)
    print("-" * 70)
    
    return outline


def save_results_to_jsonl(results: list, output_path: str):
    """
    Persists test results into a JSONL file for subsequent content generation phases.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            # Ensure non-ASCII characters (like UTF-8 text) are preserved correctly
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[Save] Results successfully saved to: {output_path}")


def main():
    print("[Main] Initializing model inference...")
    model, tokenizer = load_model()
    
    print(f"\n[Test] Launching batch inference for {len(TEST_TOPICS)} topics.\n")
    
    results = []
    for i, topic in enumerate(TEST_TOPICS, 1):
        outline = run_test(model, tokenizer, topic, i)
        results.append({
            "id": i,
            "topic": topic,
            "outline": outline,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Save the output to a file that can be used by the Content Generation module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "outline.jsonl")
    save_results_to_jsonl(results, output_path)
    
    print(f"\n{'='*70}")
    print(f"Inference Complete! Processed {len(TEST_TOPICS)} topics.")
    print(f"Output File: {output_path}")
    print('='*70)


if __name__ == "__main__":
    main()