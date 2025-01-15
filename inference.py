from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import threading
import json
from datetime import datetime

# Custom prompt tokens (matching training script)
INPUT_TOKEN = "<<INPUT>>"
OUTPUT_TOKEN = "<<OUTPUT>>"
END_TOKEN = "<eos>"

def log_request(data):
    try:
        with open("requests_log.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Error logging request: {e}")


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for cross-origin requests

# Model setup
adapter_dir = r"C:\Users\lando\Desktop\AI\LandonGPT\savedmodel"
base_model_dir = "meta-llama/Llama-3.2-3B-Instruct"

# 4-bit quantization configuration
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
    use_fast=False
)

# Match special tokens from training
special_tokens = {
    'additional_special_tokens': [INPUT_TOKEN, OUTPUT_TOKEN],
    'eos_token': END_TOKEN
}
tokenizer.add_special_tokens(special_tokens)

# Set pad_token_id if not already set
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
model = PeftModel.from_pretrained(model, adapter_dir, device_map="auto", load_in_4bit=True, is_local=True)

# Switch to eval mode
model.eval()

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route("/generate", methods=["POST"])
def generate():
    # Get user input from POST request
    data = request.json
    user_input = data.get("text", "")
    
    if not user_input:
        return jsonify({"error": "No input text provided"}), 400

    # Prepare input for the model
    model_input_text = f"{INPUT_TOKEN} {user_input} {OUTPUT_TOKEN} "
    inputs = tokenizer(model_input_text, return_tensors="pt").to(device)

    # Dynamic length constraints
    input_length = inputs["input_ids"].shape[1]
    max_response_length = min(512, input_length * 6)

    # Initialize inputs for incremental token generation
    generated_ids = inputs["input_ids"]
    generated_text = ""

    # Generate tokens step by step
    for _ in range(max_response_length):
        outputs = model.generate(
            input_ids=generated_ids,
            max_new_tokens=1,
            temperature=0.6,
            top_p=0.9,
            top_k=27,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Get the latest generated token
        new_token_id = outputs[0, -1].unsqueeze(0)

        # Decode the new token and add it to the response
        new_token = tokenizer.decode(new_token_id, skip_special_tokens=False)
        generated_text += new_token

        # Update the generated IDs with the new token
        generated_ids = torch.cat((generated_ids, new_token_id.unsqueeze(0)), dim=-1)

        # Stop if the end-of-sequence token is generated
        if new_token.strip() in [END_TOKEN, tokenizer.eos_token, "DONE"]:
            break

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input_text": user_input,
        "output_text": generated_text.strip(),
    }
    threading.Thread(target=log_request, args=(log_data,)).start()
    return jsonify({"response": generated_text.strip()})


if __name__ == "__main__":
    app.run(port=5000)  # Run the Flask app on port 5000
