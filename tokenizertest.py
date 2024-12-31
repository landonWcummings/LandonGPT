from transformers import LlamaTokenizer

# Initialize the tokenizer (replace 'meta-llama/Llama-3.2-3B-Instruct' with your model)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Define special tokens
INPUT = "<<INPUT>>"
OUTPUT = "<<OUTPUT>>"
END = "<<END>>"

# Add custom special tokens
special_tokens = {"additional_special_tokens": [INPUT, OUTPUT, END]}
tokenizer.add_special_tokens(special_tokens)

# Example input and output
example_input = "tell me a secret"
example_output = "I won’t tell you!"

# Construct the prompt
text_prompt = f"{INPUT} {example_input} {OUTPUT} {example_output} {END}"

# Tokenize
tokens = tokenizer(text_prompt, return_tensors="pt")

# Convert tokens to IDs and back to tokens
token_ids = tokens["input_ids"].squeeze().tolist()
tokens_decoded = tokenizer.convert_ids_to_tokens(token_ids)

# Print results
print("Text Prompt:", text_prompt)
print("Token IDs:", token_ids)
print("Decoded Tokens:", tokens_decoded)
