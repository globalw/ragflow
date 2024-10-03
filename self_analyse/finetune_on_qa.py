import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length

        # Load data from jsonl file
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                if 'input' in sample and 'output' in sample:
                    self.data.append(sample)
                else:
                    print(f"Skipping invalid sample: {line}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_text = sample['input']
        output_text = sample['output']

        # Concatenate input and output with a separator
        # For LLaMA-like models, ensure the separator is compatible
        sep_token = self.tokenizer.eos_token or '</s>'  # Use EOS token as separator

        # Create the combined text
        combined_text = f"{input_text}{sep_token}{output_text}{sep_token}"

        # Tokenize the combined text
        tokenized = self.tokenizer(
            combined_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()

        # Create labels by shifting input_ids; in causal LM, labels are the same as input_ids
        labels = input_ids.clone()

        # Optionally, mask the input part to focus on the output during training
        input_len = len(
            self.tokenizer(
                f"{input_text}{sep_token}", add_special_tokens=False
            )['input_ids']
        )
        labels[:input_len] = -100  # Ignore the input part in loss computation

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

class HDTFineTuneGPT2:
    def __init__(self, model_name='openlm-research/open_llama_3b', tokenizer_name=None):
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def fine_tune(
        self,
        train_dataset,
        output_dir='./fine_tuned_model',
        epochs=3,
        batch_size=2,
        lr=5e-5,
    ):
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Adjust based on your GPU memory
            learning_rate=lr,
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
            save_steps=500,
            logging_steps=100,
            evaluation_strategy='no',
            dataloader_num_workers=4,
            disable_tqdm=False,
            report_to="none",  # Set to "wandb" or "tensorboard" if using
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Start fine-tuning
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved at {output_dir}")

# Usage example
if __name__ == '__main__':
    # Path to your .jsonl file
    data_path = 'training_data/text_questions_step_answers_7870.jsonl'  # Replace with your actual file path

    # Specify the model you want to fine-tune
    model_name = 'openlm-research/open_llama_3b'  # Replace with your desired model

    # Initialize the fine-tuning class
    hdt_finetune_gpt = HDTFineTuneGPT2(model_name=model_name)
    tokenizer = hdt_finetune_gpt.tokenizer

    # Create the dataset
    train_dataset = CustomDataset(tokenizer, data_path)

    # Fine-tune the model
    hdt_finetune_gpt.fine_tune(
        train_dataset,
        output_dir='./fine_tuned_model',
        epochs=3,
        batch_size=2,  # Adjust based on your GPU capacity
        lr=5e-5,
    )
