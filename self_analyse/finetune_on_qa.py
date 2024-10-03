import warnings
warnings.filterwarnings("ignore")


import json
import torch
import logging
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training



# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=256):  # Reduced max_length
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length

        # Load data from jsonl file
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                if 'question' in sample and 'answer' in sample:
                    question = sample['question'].strip()
                    answer = sample['answer'].strip()
                    if question and answer:
                        self.data.append(sample)
                    else:
                        logger.debug(f"Skipping empty sample: {line}")
                else:
                    logger.debug(f"Skipping invalid sample: {line}")

        # Check the length of the dataset
        logger.info(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question_text = sample['question']
        answer_text = sample['answer']

        # Concatenate question and answer with a separator
        sep_token = self.tokenizer.eos_token or '</s>'  # Use EOS token as separator

        # Create the combined text
        combined_text = f"{question_text}{sep_token}{answer_text}{sep_token}"

        # Tokenize the combined text
        tokenized = self.tokenizer(
            combined_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        logger.info(f"Tokenized input: {tokenized}")
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()
        logger.info(f"Input IDs: {input_ids}")
        # Create labels by copying input_ids
        labels = input_ids.clone()

        # Mask the question part to focus on the answer during training
        question_len = len(
            self.tokenizer(
                f"{question_text}{sep_token}", add_special_tokens=False
            )['input_ids']
        )
        labels[:question_len] = -100  # Ignore the question part in loss computation
        logger.info(f"Labels: {labels}")
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class HDTFineTuneGPT2:
    def __init__(self, model_name='gpt2', tokenizer_name=None):
        if tokenizer_name is None:
            tokenizer_name = model_name

        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        logger.info(f"Quantization config: {quantization_config}")
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map='auto',
        )
        logger.info(f"Loaded model: {model_name}")
        # Set use_cache to False
        self.model.config.use_cache = False

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],  # Adjust based on model architecture
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        logger.info("LoRA applied to the model")

        # Enable gradient checkpointing with use_reentrant=False
        self.model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
        logger.info("Model and tokenizer loaded")

    def fine_tune(
        self,
        train_dataset,
        output_dir='./fine_tuned_model',
        epochs=1,  # Reduced epochs for testing
        batch_size=1,
        lr=2e-5,  # Adjusted learning rate
        max_steps=500,
    ):
        logger.info("Starting fine-tuning process")

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            learning_rate=lr,
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            save_total_limit=2,
            save_steps=100,
            logging_steps=10,
            logging_strategy='steps',
            evaluation_strategy='no',
            dataloader_num_workers=2,
            disable_tqdm=False,
            report_to="none",
            load_best_model_at_end=False,
            local_rank=-1,
            log_level='info',
            log_on_each_node=False,
            per_device_eval_batch_size=1,
            dataloader_pin_memory=True,
        )
        logger.info("Training arguments set")
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )
        logger.info("Data collator created")
        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[MemoryMonitorCallback()],
        )
        logger.info("Trainer initialized")
        # Start fine-tuning
        logger.info("Starting training")
        trainer.train()

        # Save the fine-tuned model
        logger.info("Training completed. Saving model")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved at {output_dir}")

class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Log GPU memory usage
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            logger.info(f"GPU memory allocated: {gpu_mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

# Usage example
if __name__ == '__main__':
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Please ensure you have a CUDA-enabled GPU.")
    else:
        logger.info("CUDA is available. Using GPU for training.")

    # Path to your .jsonl file
    data_path = 'training_data/text_questions_step_answers_7870.jsonl'  # Replace with your actual file path

    # Specify the model you want to fine-tune
    model_name = 'gpt2'  # Replace with your desired model

    # Initialize the fine-tuning class
    hdt_finetune_gpt = HDTFineTuneGPT2(model_name=model_name)
    tokenizer = hdt_finetune_gpt.tokenizer

    # Create the dataset
    train_dataset = CustomDataset(tokenizer, data_path, max_length=256)

    # Fine-tune the model
    hdt_finetune_gpt.fine_tune(
        train_dataset,
        output_dir='./fine_tuned_model',
        epochs=1,
        batch_size=6,
        lr=2e-5,
    )
