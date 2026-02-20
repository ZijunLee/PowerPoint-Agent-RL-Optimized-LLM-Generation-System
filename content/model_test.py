import os
import json
import time
import re
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from prompt import CONTENT_SYSTEM_PROMPT, CONTENT_USER_PROMPT

# Load backend/.env configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
dotenv.load_dotenv(os.path.join(BACKEND_DIR, '.env'))

# ---------- Configuration ----------
BASE_MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_CACHE_DIR = "/data/xxx/model" # Replace with your local model cache directory

# Path to your fine-tuned content generation model checkpoint
TRAINED_MODEL_PATH = os.path.join(SCRIPT_DIR, "output/content01-20251221-093206")


# ============================================================
# TEST DATA (3 Sample Outlines for Content Expansion)
# ============================================================
TEST_OUTLINES = [
    {
        "topic": "Applications of AI in Medical Diagnosis",
        "outline": """# Applications of AI in Medical Diagnosis

## Technical Foundations and Current Status
### Core Algorithms and Architectures
- Implement Deep CNNs for medical image recognition
- Apply Transformer architectures for electronic health record (EHR) processing
- Design multi-modal fusion models to integrate diverse medical data
- Optimize inference speed to meet real-time diagnostic requirements

### Data Collection and Processing Pipeline
- Establish standardized medical imaging data collection protocols
- Implement data de-identification and privacy protection mechanisms
- Design high-efficiency data annotation quality control systems

## Clinical Application Scenarios
### Medical Imaging Assisted Diagnosis
- Enable automated lesion detection in lung CT scans
- Build screening systems for diabetic retinopathy in fundus images
- Develop intelligent analysis tools for mammograms

### Intelligent Pathology Analysis
- Apply AI to identify morphological features of tumor cells
- Automate the generation of pathological grading reports
- Predict tumor molecular subtypes and prognosis

## Challenges and Future Trends
### Technical Bottlenecks and Solutions
- Improve model generalization in small-sample scenarios
- Enhance model interpretability to build physician trust
- Resolve cross-center data distribution discrepancy issues

### Future Directions
- Drive the construction of multi-center federated learning platforms
- Explore the potential of LLMs in the medical domain
- Establish regulatory frameworks for AI-assisted diagnosis
"""
    },
    {
        "topic": "New Energy Vehicle (NEV) Industry Chain Analysis",
        "outline": """# New Energy Vehicle (NEV) Industry Chain Analysis

## Upstream Core Segments
### Battery Technology Evolution
- Analyze technical differences between LFP and NCM batteries
- Evaluate commercialization progress and challenges of solid-state batteries
- Discuss technical paths for improving battery energy density
- Study the impact of fast-charging technology on battery performance

### Key Raw Material Supply Landscape
- Map global distribution of lithium resources and supply chain risks
- Analyze price volatility factors for cobalt, nickel, and other materials
- Evaluate economic feasibility of battery recycling and reuse

## Midstream Manufacturing Segments
### Manufacturing Process Innovation
- Promote integrated die-casting technology to reduce production costs
- Optimize battery pack integration design to improve space utilization
- Implement intelligent production lines to increase manufacturing efficiency

### Core Component Localization
- Accelerate independent R&D of core electric drive system technologies
- Advance the localization process for automotive-grade chips
- Cultivate domestic suppliers for thermal management systems

## Downstream Application Scenarios
### Charging Infrastructure Development
- Plan urban public charging network layouts
- Promote construction standards for supercharging stations
- Explore commercial feasibility of battery swapping models
- Develop Vehicle-to-Grid (V2G) technology

### Market Trend Forecasting
- Forecast NEV penetration rates for 2025
- Analyze growth potential in lower-tier markets
- Evaluate competitive landscape in export markets
"""
    },
    {
        "topic": "Strategies for Enhancing Remote Work Efficiency",
        "outline": """# Strategies for Enhancing Remote Work Efficiency

## Work Environment and Infrastructure
### Home Office Space Optimization
- Design ergonomic layouts for home office areas
- Configure professional-grade video conferencing equipment and networks
- Establish physical boundaries between work and personal life
- Optimize lighting and ventilation for increased comfort

### Digital Toolchain Integration
- Deploy enterprise-grade collaborative office platforms
- Integrate project management and task tracking systems
- Build secure and reliable VPN access solutions

## Time Management and Self-Motivation
### High-Efficiency Work Rhythm Design
- Implement the Pomodoro Technique to improve focus
- Schedule fixed daily slots for deep work
- Set clear daily goals and priorities

### Self-Supervision Mechanisms
- Use time-tracking tools to quantify work efficiency
- Establish weekly work review and improvement cycles
- Cultivate self-discipline and intrinsic motivation
- Design reasonable rest and recovery strategies

## Team Collaboration and Communication
### Asynchronous Communication Best Practices
- Define clear written communication standards and templates
- Use asynchronous messaging properly to minimize interruptions
- Build a document-driven knowledge sharing system

### Team Cohesion Maintenance
- Organize regular virtual team-building events to enhance belonging
- Implement one-on-one remote mentorship programs
- Establish optimal time windows for cross-timezone collaboration
- Create virtual spaces for informal social interaction
"""
    }
]


def load_model():
    """Loads the fine-tuned content generation model and tokenizer."""
    model_path = TRAINED_MODEL_PATH
    
    if os.path.exists(model_path):
        print(f"[Model] Loading fine-tuned checkpoint: {model_path}")
    else:
        print(f"[Model] Path not found: {model_path}. Falling back to base model: {BASE_MODEL_NAME}")
        model_path = BASE_MODEL_NAME
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        cache_dir=MODEL_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with auto-device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16
    )
    
    # Hardware monitoring
    if hasattr(model, 'hf_device_map'):
        print(f"[GPU] Device Map Distribution: {model.hf_device_map}")
    else:
        print(f"[GPU] Main Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def generate_full_content(model, tokenizer, topic: str, outline: str) -> str:
    """Expands a given outline into detailed, structured content."""
    # Build prompt messages
    messages = [
        {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
        {"role": "user", "content": CONTENT_USER_PROMPT.format(
            topic=topic,
            outline=outline
        )}
    ]
    
    # Apply ChatML template
    if hasattr(tokenizer, 'apply_chat_template'):
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback template
        input_text = f"System: {CONTENT_SYSTEM_PROMPT}\n\nUser: Topic: {topic}\nOutline: {outline}\n\nAssistant:"
    
    # Move inputs to active GPU
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Perform generation (allowing for long-form output)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,  # Supports long document generation
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and exclude the input prompt
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return content.strip()


def run_test(model, tokenizer, outline_data: dict, index: int, total: int):
    """Executes inference for a single test case and previews the output."""
    topic = outline_data["topic"]
    outline = outline_data["outline"]
    
    print(f"\n{'='*70}")
    print(f"Test Case {index}/{total}")
    print(f"Topic: {topic}")
    print(f"Outline Length: {len(outline)} characters")
    print('='*70)
    
    content = generate_full_content(model, tokenizer, topic, outline)
    
    print("\nGenerated Full Content Preview:")
    print("-" * 70)
    # Preview the first 2000 characters
    print(content[:2000] + "..." if len(content) > 2000 else content)
    print("-" * 70)
    print(f"Total Content Length: {len(content)} characters")
    
    return content


def save_results_to_jsonl(results: list, output_path: str):
    """Saves the final generated documents to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[Save] Results successfully saved to: {output_path}")


def main():
    print("[Main] Starting model initialization...")
    model, tokenizer = load_model()
    
    # Launch batch inference for sample outlines
    print(f"\n[Test] Processing {len(TEST_OUTLINES)} sample outlines.\n")
    
    results = []
    for i, outline_data in enumerate(TEST_OUTLINES, 1):
        content = run_test(model, tokenizer, outline_data, i, len(TEST_OUTLINES))
        results.append({
            "id": i,
            "topic": outline_data["topic"],
            "outline": outline_data["outline"],
            "generated_content": content,
            "content_length": len(content),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Persistence
    output_path = os.path.join(SCRIPT_DIR, "content.jsonl")
    save_results_to_jsonl(results, output_path)
    
    print(f"\n{'='*70}")
    print(f"Inference Complete! Generated content for {len(TEST_OUTLINES)} outlines.")
    print(f"Results File: {output_path}")
    print('='*70)


if __name__ == "__main__":
    main()