import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# Custom prompt tokens (matching training script)
INPUT_TOKEN = "<<INPUT>>"
OUTPUT_TOKEN = "<<OUTPUT>>"
END_TOKEN = "<eos>"

# Directories
base_model_dir = "meta-llama/Llama-3.2-3B-Instruct"  # Base model ID or local path
adapter_dir = "savedmodel"                         # Directory containing LoRA adapter

# 4-bit quantization configuration (same as in training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_dir,
    padding_side="left",
    use_fast=False  # LLaMA often needs use_fast=False
)

# Match special tokens from training
special_tokens = {
    'additional_special_tokens': [INPUT_TOKEN, OUTPUT_TOKEN],
    'eos_token': END_TOKEN  # We link <eos> string to the eos_token_id
}
tokenizer.add_special_tokens(special_tokens)

# In LLaMA, eos_token_id=2 by default, so set pad_token_id accordingly
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# Resize embeddings for new special tokens
model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_dir, device_map="auto")

# Switch to eval mode
model.eval()

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

################################################################################
# LOOP FOR COMMAND LINE INTERACTION
################################################################################

print("Model is ready! Type your input below and press Enter. Press Ctrl+C to exit.\n")

try:
    while True:
        # Prompt user for input
        textinput = input(">> ")

        # Prepare input for the model
        model_input_text = f"{INPUT_TOKEN} {textinput} {OUTPUT_TOKEN} "
        inputs = tokenizer(model_input_text, return_tensors="pt").to(device)

        # Dynamic length constraints
        input_length = inputs["input_ids"].shape[1]
        max_response_length = min(512, input_length * 6)
        min_length = max(10, input_length)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_length=max_response_length,
            min_length=min_length,
            temperature=0.6,
            top_p=0.9,
            top_k=27,
            no_repeat_ngram_size=2
        )

        # Decode and print the model's response
        for seq in outputs:
            decoded = tokenizer.decode(seq, skip_special_tokens=False)
            print(decoded)

except KeyboardInterrupt:
    print("\nExiting. Goodbye!")
