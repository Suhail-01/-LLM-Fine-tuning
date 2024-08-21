# llm_fine_tuning.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load the tokenizer and model
model_name = "gpt2"  # Replace with the model you want to fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and preprocess dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def main():
    # Paths to your dataset files
    train_path = "./data/train.txt"
    val_path = "./data/val.txt"

    # Load datasets
    train_dataset = load_dataset(train_path, tokenizer)
    val_dataset = load_dataset(val_path, tokenizer)

    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # MLM=False for causal language modeling (like GPT)
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=200,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    main()
