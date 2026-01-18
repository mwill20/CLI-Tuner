from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

print("Loading adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "models/checkpoints/phase2-final"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

print("✅ Model loads successfully!")
print(f"Model device: {model.device}")
