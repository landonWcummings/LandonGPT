from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

input = "<<INPUT>>"
output = "<<OUTPUT>>"
eom = "<<END>>"
messagesep = "|"

# Paths
base_model_dir = "meta-llama/Llama-3.2-3B-Instruct"  # Base model directory
adapter_dir = "saved-model"  # Directory containing LoRA adapter

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)

special_tokens = {
    'additional_special_tokens': [input, output, eom]
}
tokenizer.add_special_tokens(special_tokens)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True)
model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_dir)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Example input, formatted like in training data
textinput = "what is the meaning of life?"  # Simulating structured input

# Extract only the `input` field
model_input = input + " " + textinput + " " + output 

# Tokenize the input for the model
inputs = tokenizer(model_input, return_tensors="pt").to(device)

eom_token_id = tokenizer.convert_tokens_to_ids(eom)

# Set dynamic length constraints
input_length = len(inputs["input_ids"][0])
max_response_length = min(256, input_length * 4)  # Maximum response length based on input size
min_length = max(10, input_length)  # Minimum response length

# List of maximum lengths to explore
max_lengths = [10, 40,70, 100]

# Generate responses for different max lengths
for maxl in max_lengths:
    if maxl <= min_length:
        maxl = min_length + 1
    output = model.generate(
        **inputs,
        max_length=maxl,
        min_length=min_length,  # Ensure a minimum response length
        num_return_sequences=2,
        eos_token_id=eom_token_id,  # Number of sequences to generate
        no_repeat_ngram_size=2,  # Avoid repetition
        temperature=0.1,  # Adjust for creativity
        top_p=0.9,        # Use nucleus sampling
        top_k=39,
        early_stopping=True         # Adjust for more creative or deterministic output
    )

    # Decode the output
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in output]
    for idx, text in enumerate(generated_texts):
        print(f"Max length {maxl} - Option {idx+1}: {text}")
