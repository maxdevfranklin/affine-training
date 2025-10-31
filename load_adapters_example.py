"""
Example: How to Load and Use Saved LoRA Adapters for Inference

This script shows how to load and use the saved adapters from RL training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model_with_adapters(base_model_path: str, adapter_path: str):
    """
    Load base model and apply saved LoRA adapters
    
    Args:
        base_model_path: Path to base model (e.g., "../Affine-QQ")
        adapter_path: Path to adapter directory (e.g., "models/rl_best")
    
    Returns:
        (model, tokenizer) tuple
    """
    # Load base model
    print(f"Loading base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load adapters
    print(f"Loading adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapters into base model (optional, makes inference faster)
    print("Merging adapters...")
    model = model.merge_and_unload()
    
    model.eval()
    print("Model ready for inference!")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate text using the model"""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Example usage
    base_model = "../Affine-QQ"
    adapter_model = "models/rl_best"  # or "models/sft_best"
    
    # Load model with adapters
    model, tokenizer = load_model_with_adapters(base_model, adapter_model)
    
    # Use for inference
    prompt = "Solve this problem: 2 + 2 = ?"
    response = generate_text(model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    
    # You can now use this model for:
    # - Inference
    # - Generation
    # - Evaluation
    # - Deployment

