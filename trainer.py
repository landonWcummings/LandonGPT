import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from huggingface_hub import login
from datasets import load_dataset
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="pt")


# Custom prompt tokens (optional)
INPUT_TOKEN = "<<INPUT>>"
OUTPUT_TOKEN = "<<OUTPUT>>"
END_TOKEN = "<eos>"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )

def generate_and_tokenize_prompt(example, tokenizer, max_length=512, verbose=False):
    """
    Tokenizes the input-output pair. 
    Now we remove the <<END>> from the final text, letting default eos token do the job.
    """
    text_prompt = (
        f"{INPUT_TOKEN} {example['input']} "
        f"{OUTPUT_TOKEN} {example['output']} DONE {tokenizer.eos_token}"
        # We do NOT add custom END_TOKEN here 
    )

    # Tokenize without padding
    tokens = tokenizer(
        text_prompt,
        truncation=True,
        max_length=max_length,
        padding="do_not_pad",
    )

    # The user portion length (so we mask out the user portion in the loss)
    user_part_len = len(tokenizer(f"{INPUT_TOKEN} {example['input']} ")["input_ids"])

    labels = [-100] * len(tokens["input_ids"])
    for i in range(user_part_len, len(tokens["input_ids"])):
        labels[i] = tokens["input_ids"][i]

    tokens["labels"] = labels

    # Turn them into tensors
    tokens = {k: torch.tensor(v) for k, v in tokens.items()}
    if (verbose):
        print(tokenizer.eos_token)
        print(tokenizer.eos_token_id)

        print(f"Text: {text_prompt}")
        print(f"labels: {labels}")
        print(f"tokens: {tokens}")
        print(f"Special Tokens: {tokenizer.special_tokens_map}")

    

    return tokens

def main(
    train_dataset_path,
    output_dir,
    eval_dataset_path=None,
    base_model_id="meta-llama/Llama-3.2-3B-Instruct",
    max_length=512,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    batch_size=1,
    epochs=5,
    gradient_accumulation_steps=4,
    use_bf16=True,
    huggingface_token=None  # Optionally pass in your HF token
):
    """
    Main fine-tuning script for LLaMA model using QLoRA and LoRA.
    """
    # Clear cache
    torch.cuda.empty_cache()

    # If you want to push to HF Hub, you can log in
    if huggingface_token:
        login(token=huggingface_token)
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load datasets
    train_dataset = load_dataset('json', data_files=train_dataset_path, split='train')
    eval_dataset = (
        load_dataset('json', data_files=eval_dataset_path, split='train')
        if eval_dataset_path
        else None
    )

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        use_fast=False,  # LLaMA often needs fast=False
        padding="longest"
    )

    # LLaMA defaults
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2  # default LLaMA end-of-sequence token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Optional: Add custom prompt tokens
    # We omit END_TOKEN from special tokens to rely on the default LLaMA eos.
    special_tokens = {
        'additional_special_tokens': [INPUT_TOKEN, OUTPUT_TOKEN],
        'eos_token': END_TOKEN  # This ensures the EOS token is linked properly
    }

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print("Beginning tests:")
    example = {"input": "Tell me a secret.", "output": "I won’t tell you!"}
    generate_and_tokenize_prompt(example, tokenizer, max_length=max_length,verbose=True)
    print("<eos token id>:", tokenizer.eos_token_id)

    time.sleep(5)

    # Map datasets with the tokenization function
    def tokenize_fn(ex):
        return generate_and_tokenize_prompt(ex, tokenizer, max_length=max_length)

    tokenized_train_dataset = train_dataset.map(tokenize_fn)
    tokenized_val_dataset = eval_dataset.map(tokenize_fn) if eval_dataset else None

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Wrap the base model in LoRA
    model = get_peft_model(model, config)

    # Print trainable parameters
    print_trainable_parameters(model)

    # FSDP plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    # Accelerator
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare_model(model)

    # For multi-GPU
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Setup Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        learning_rate=2.5e-5,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        logging_steps=50,
        logging_dir="./logs",
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Ensure LLaMA doesn't cache (important for LoRA)
    model.config.use_cache = False

    # Train
    trainer.train()

    # Save final model
    trainer.save_model()

if __name__ == "__main__":
    main(
        train_dataset_path=r"C:\Users\lndnc\OneDrive\Desktop\AI test\landongpt\data\gpttrain.jsonl",
        output_dir="savedmodel",
        eval_dataset_path=None,
        base_model_id="meta-llama/Llama-3.2-3B-Instruct",
        epochs=3,
        gradient_accumulation_steps=2,
        huggingface_token="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"  # or "hf_yourTokenHere"
    )
