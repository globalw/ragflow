import os
import re
import json
import torch
import logging
from transformers import BloomForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset, default_collate


class CustomDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        description, comment, label = self.pairs[idx]

        # Check for empty description or comment
        if not description.strip() or not comment.strip():
            logging.warning(f"Empty description or comment at index {idx}. Skipping entry.")
            return None  # Skip invalid entries

        # Tokenize inputs
        inputs = self.tokenizer(description, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        labels = self.tokenizer(comment, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids

        inputs["labels"] = labels.squeeze()  # Labels need to be in the correct format
        return {key: val.squeeze(0) for key, val in inputs.items()}


class HDTFineTuneGPT2:
    def __init__(self, pdf_directory='./jira', output_file='./jira/merged_hdt_content.txt',
                 do_extraction=False, process_textfile=False, verbose=True, training_pairs=[], model_name="bigscience/bloom-560m"):
        self.do_extraction = do_extraction
        self.process_textfile = process_textfile
        self.verbose = verbose
        self.training_pairs = training_pairs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BloomForCausalLM.from_pretrained(model_name)
        self.limit_gpu_memory(max_gb=10)
        if self.do_extraction:
            self.extract_text_from_pdfs(pdf_directory, output_file)
        if self.process_textfile:
            self.process_text_file(input_file_path=output_file, output_file_path='./jira/augmented_hdt_content.txt')

    # Set the memory usage limit to 10 GB (on a 16 GB GPU, this would be about 0.625)
    def limit_gpu_memory(self, max_gb=10):
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Total memory in GB
        fraction = min(max_gb / total_mem, 1.0)  # Compute fraction of GPU memory to limit
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)

    def custom_collate(self, batch):
        batch = [b for b in batch if b is not None]
        return default_collate(batch)

    def read_text_sections(self, input_file_path, delimiter="=====NEXT====="):
        with open(input_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Split the content into sections using the delimiter
        sections = content.split(delimiter)
        # Clean up the sections by stripping whitespace
        text_sections = [section.strip() for section in sections if section.strip()]
        return text_sections

    def run_inference(self, prompt, max_new_tokens=50):
        self.limit_gpu_memory(max_gb=10)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.eval()

        # Tokenize the input with truncation and return the attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Move inputs to the correct device (GPU/CPU)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            # Generate text with the model
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Adjust this parameter
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                num_beams=5,
                early_stopping=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate_question(self, text_section):
        # Check the length of the input text
        print(f"Generating question for text section (length: {len(text_section)} tokens)")

        # Prepare the prompt for question generation
        prompt = f"Generate a question based on the following text:\n\n{text_section}\n\nQuestion:"

        # Run inference to generate the question
        generated_text = self.run_inference(prompt, max_new_tokens=50)  # Use max_new_tokens instead of max_length

        # Extract the question from the generated text
        question_match = re.search(r'Question:(.*)', generated_text, re.IGNORECASE)
        if question_match:
            question = question_match.group(1).strip()
        else:
            question = generated_text.strip()
        return question

    def create_text_sections_and_questions(self, input_file_path):
        # Read the text sections from the file
        text_sections = self.read_text_sections(input_file_path)
        questions = []
        for idx, section in enumerate(text_sections):
            print(f"Generating question for section {idx + 1}/{len(text_sections)}")
            question = self.generate_question(section)
            questions.append(question)
        return text_sections, questions

    def save_text_sections_and_questions(self, text_sections, questions, output_file_path="text_questions.jsonl"):
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for text_section, question in zip(text_sections, questions):
                entry = {
                    "text_section": text_section,
                    "question": question
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        print(f"Text sections and questions saved to {output_file_path}")

    def process_text_file(self, input_file_path, output_file_path):
        # Implement your logic to process the text file here
        print(f"Processing text file from {input_file_path} to {output_file_path}")
        # Example output
        self.training_pairs = [('Description1', 'Comment1', 1), ('Description2', 'Comment2', 1)]  # Sample data

    def execute_pipeline(self, model_name='fine_tuned_model', model_output_dir='fine_tuned_gpt2_model',
                         prompt="Paraphrasiere diesen Text"):
        # Process the text file and generate interpretations
        if self.process_textfile:
            self.process_text_file(input_file_path='./jira/merged_hdt_content.txt', output_file_path='./jira/augmented_hdt_content.txt')

        # Create text sections and questions
        text_sections, questions = self.create_text_sections_and_questions(input_file_path='./jira/augmented_hdt_content.txt')

        # Optionally, save the text sections and questions
        self.save_text_sections_and_questions(text_sections, questions, output_file_path="text_questions.jsonl")

        # The rest of your pipeline can continue here if needed
        # For example, fine-tuning, quantization, building models, etc.


# Run the pipeline
if __name__ == '__main__':
    hdt_finetune_gpt = HDTFineTuneGPT2(do_extraction=False, process_textfile=True)
    hdt_finetune_gpt.execute_pipeline()
