import os
import shutil
import subprocess
import requests
import PyPDF2
import re
import logging
import torch
import json
from transformers import (BloomForCausalLM, AutoTokenizer, T5ForConditionalGeneration,
                          T5Tokenizer, MBartForConditionalGeneration, MBart50TokenizerFast)
from torch.amp import GradScaler  # Use the updated torch.amp API
from torch.utils.data import DataLoader, default_collate, Dataset
from torch.optim import AdamW
from concurrent.futures import ThreadPoolExecutor, as_completed


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

        if self.do_extraction:
            self.extract_text_from_pdfs(pdf_directory, output_file)
        if self.process_textfile:
            self.process_text_file(input_file_path=output_file, output_file_path='./jira/augmented_hdt_content.txt')

    def custom_collate(self, batch):
        batch = [b for b in batch if b is not None]
        return default_collate(batch)

    def finetune_gpt(self, output_model_dir='./fine_tuned_model', epochs=3, learning_rate=5e-5,
                     accumulation_steps=4, patience_steps=10, min_delta=1e-4):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        dataset = CustomDataset(self.training_pairs, self.tokenizer)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=16, collate_fn=self.custom_collate, num_workers=4,
                                pin_memory=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scaler = GradScaler()

        self.model.train()
        optimizer.zero_grad()
        for epoch in range(epochs):
            epoch_loss = 0
            best_loss = float('inf')
            steps_no_improve = 0
            for step, batch in enumerate(dataloader):
                batch = {key: val.to(device) for key, val in batch.items()}
                with torch.cuda.amp.autocast('cuda'):
                    outputs = self.model(**batch)
                    loss = outputs.loss / accumulation_steps
                    epoch_loss += loss.item()
                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if loss.item() < best_loss - min_delta:
                    best_loss = loss.item()
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1
                if steps_no_improve >= patience_steps:
                    print(f"Breaking step loop at step {step} in epoch {epoch} due to no improvement.")
                    break
            epoch_loss /= len(dataloader)
            print(f"Epoch {epoch} completed with average loss: {epoch_loss}")
        self.model.save_pretrained(output_model_dir)
        self.tokenizer.save_pretrained(output_model_dir)
        print(f"Model saved at {output_model_dir}")


    def run_inference(self, prompt):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.eval()

        # Tokenize the input with padding and return the attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Move inputs to the correct device (GPU/CPU)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            # Pass the attention mask along with the input_ids
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_length=100,
                                          num_return_sequences=1)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_text_from_pdfs(self, pdf_dir, output_txt_file):
        logging.basicConfig(filename='pdf_extraction.log', level=logging.INFO)
        with open(output_txt_file, 'w', encoding='utf-8') as output_file:
            for filename in os.listdir(pdf_dir):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(pdf_dir, filename)
                    try:
                        with open(pdf_path, 'rb') as pdf_file:
                            reader = PyPDF2.PdfReader(pdf_file)
                            text = ''
                            for page_num in range(len(reader.pages)):
                                page = reader.pages[page_num]
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text
                                else:
                                    logging.warning(f"No text extracted from page {page_num} in {filename}")
                            if text.strip():
                                print(f"Extracted text from {filename}")
                                output_file.write(f"\n=== {filename} ===\n")
                                output_file.write(text + "\n\n")
                            else:
                                logging.warning(f"No text extracted from {filename}")
                    except Exception as e:
                        logging.error(f"Error processing {filename}: {e}")

    def generate_interpretation_hf(self, description, comment, src_lang="de_DE"):
        # Load pre-trained mBART model and tokenizer
        model_name = 'facebook/mbart-large-50'
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)

        # Load model with GPU support
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Set the tokenizer source language
        tokenizer.src_lang = src_lang

        # Construct the input prompt
        prompt = f"Gegeben ist die folgende Beschreibung und der Kommentar. Was war die Anfrage und wie war der Lösungsweg, wenn kein Lösungsweg erkennbar ist, sei der Lösungsweg nicht vorhanden:\n\nBeschreibung: {description}\n\nKommentar: {comment}\n\nInterpretation:"

        # Encode the input and generate output
        inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,  # Control the number of new tokens generated
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode the output
        interpretation = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return interpretation

    def parse_llama_response(self, response_content):
        # Split the content by newlines to get individual JSON objects
        lines = response_content.split(b'\n')

        # Extract and concatenate the 'response' fields
        full_response = ""
        for line in lines:
            if line.strip():  # Skip empty lines
                try:
                    # Parse each line as JSON
                    response_data = json.loads(line)
                    # Append the response text to full_response
                    full_response += response_data.get('response', '')
                except json.JSONDecodeError:
                    print("Error decoding JSON:", line)

        return full_response

    def generate_interpretation_llama(self, description, comment):
        # Prepare the input prompt in German
        prompt = f"Gegeben ist die folgende Beschreibung und der Kommentar. Geben Sie eine Interpretation:\n\nBeschreibung: {description}\n\nKommentar: {comment}\n\nInterpretation:"

        # Make a request to Ollama's API
        response = requests.post(
            "http://localhost:11434/api/generate",  # Example URL, replace with the correct one if different
            json={"model": "llama3.1:8b",
                  "prompt": prompt}
        )

        # Check if the request was successful
        if response.status_code == 200:
            response_content = response.content
            # Parse the streamed response to get the full interpretation
            interpretation = self.parse_llama_response(response_content)
            return interpretation
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    def split_description_comments(self, text):
        pairs = []
        tickets = re.split(r'=== HDT-\d+\.pdf ===', text)

        # Function to process each comment and generate interpretation
        def process_description_and_comment(description, comment):
            interpretation = self.generate_interpretation_llama(description, comment.strip())
            return (description, comment.strip(), interpretation, 1)

        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your CPU/GPU capabilities
            future_to_comment = {}
            for ticket in tickets:
                if not ticket.strip():
                    continue
                description_match = re.search(r'Description of HDT-\d+:([\s\S]*?)(?=Comments:|$)', ticket)
                description = description_match.group(1).strip() if description_match else ""

                comments_match = re.search(r'Comments:([\s\S]*)', ticket)
                comments = comments_match.group(1).strip() if comments_match else ""

                # Skip this ticket if the comment is empty
                if not comments:
                    continue

                individual_comments = re.split(r'Comment by .*?:', comments)
                for comment in individual_comments:
                    if comment.strip():  # Only consider non-empty comments
                        # Submit the task to the executor
                        future = executor.submit(process_description_and_comment, description, comment.strip())
                        future_to_comment[future] = (description, comment.strip())

            # Collect the results as they complete
            for future in as_completed(future_to_comment):
                try:
                    result = future.result()
                    pairs.append(result)
                    print(f"Processed comment: {future_to_comment[future][1]}")
                except Exception as exc:
                    description, comment = future_to_comment[future]
                    print(f"An error occurred while processing the comment '{comment}': {exc}")

        return pairs

    def save_pairs_to_file(self, output_file_path, delimiter="=====NEXT====="):
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for description, comment, interpretation, label in self.training_pairs:
                # Format the data into a single string
                formatted_entry = f"Description:\n{description}\n\nComment:\n{comment}\n\nInterpretation:\n{interpretation}\n"

                # Write the formatted entry to the file
                file.write(formatted_entry)

                # Write the delimiter
                file.write(f"\n{delimiter}\n")


    def process_text_file(self, input_file_path, output_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        self.training_pairs = self.split_description_comments(file_content)
        self.save_pairs_to_file(output_file_path)

        for i, pair in enumerate(self.training_pairs):
            print(f"Pair {i + 1}:")
            print(f"Description: {pair[0]}")
            print(f"Comment: {pair[1]}")
            print(f"Label: {pair[2]}")
            print("\n")

    def quantize_model(self, model_output_dir='./fine_tuned_gpt2_model'):
        try:
            # Define the Docker command to run the quantization step
            docker_command = [
                "docker", "run", "--rm",
                "-v", f"{os.path.abspath(model_output_dir)}:/model",
                "ollama/quantize", "-q", "q4_K_M", "/model"
            ]
            # Execute the Docker command
            subprocess.run(docker_command, check=True)
            print(f"Model quantized successfully and saved at {model_output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error during model quantization: {e}")

    from transformers import T5ForConditionalGeneration, T5Tokenizer

    def paraphrase_text(self, text, num_return_sequences=3, max_length=256):
        model_name = "ramsrigouthamg/t5_paraphraser"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Prepare the text for T5
        input_text = f"paraphrase: {text} </s>"
        encoding = tokenizer.encode_plus(input_text, max_length=max_length,
                                         padding="max_length", return_tensors="pt",
                                         truncation=True)

        # Generate paraphrases
        outputs = model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            num_beams=10,
            temperature=1.5,
            repetition_penalty=2.0,
            early_stopping=True
        )

        # Decode the paraphrased sentences
        paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                             output in outputs]

        return paraphrased_texts


    def create_modelfile(self, model_name, model_path):
        # Replace backslashes with forward slashes for compatibility
        model_path = model_path.replace("\\", "/")

        modelfile_content = f"""
        FROM {model_path}
        PARAMETER stop "<|im_start|>"
        PARAMETER stop "<|im_end|>"
        TEMPLATE \"\"\"
        <|im_start|>system
        {{ .System }}<|im_end|>
        <|im_start|>user
        {{ .Prompt }}<|im_end|>
        <|im_start|>assistant
        \"\"\"
        """
        modelfile_path = os.path.join(model_name, 'Modelfile')
        os.makedirs(model_name, exist_ok=True)
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        print(f"Modelfile created at {modelfile_path}")
        return modelfile_path

    def build_ollama_model(self, model_name):
        try:
            # Use the `ollama create` command to register the model
            subprocess.run(['ollama', 'create', model_name, '-f', os.path.join(os.getcwd(), model_name, 'Modelfile')],
                           check=True)
            print(f"Model '{model_name}' successfully built in Ollama.")
        except subprocess.CalledProcessError as e:
            print(f"Error building the model: {e}")

    def ensure_quantization_file_structure(self, model_dir):
        """
        Ensures the correct file structure is present for quantizing a model using Docker.
        This includes checking for config.json, model.safetensors, and tokenizer files.

        Args:
            model_dir (str): The path to the model directory where the quantization will happen.
        """
        # Required files for quantization
        required_files = [
            'config.json',  # Model config
            'model.safetensors',  # Model weights
            'tokenizer.json',  # Tokenizer JSON file
            'tokenizer_config.json'  # Tokenizer configuration
        ]

        # Alternative file names that can be renamed if missing required files
        alternative_files = {
            'config.sentence_transformers.json': 'config.json',
            # Add more alternative mappings if needed.
        }

        # Check if the model directory exists
        if not os.path.isdir(model_dir):
            print(f"Error: The model directory '{model_dir}' does not exist.")
            return False

        # Check for required files
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.isfile(file_path):
                missing_files.append(file)

        # Handle missing files by renaming alternatives, if possible
        for alt_file, required_file in alternative_files.items():
            alt_file_path = os.path.join(model_dir, alt_file)
            required_file_path = os.path.join(model_dir, required_file)
            if required_file in missing_files and os.path.isfile(alt_file_path):
                # Rename or copy the alternative file to the required file
                print(f"Renaming {alt_file} to {required_file}")
                shutil.copy(alt_file_path, required_file_path)
                missing_files.remove(required_file)

        # Final check for any remaining missing files
        if missing_files:
            print(f"Error: The following required files are missing in '{model_dir}': {', '.join(missing_files)}")
            return False

        print(f"All required files are present in '{model_dir}'.")
        return True

    def run_ollama_model(self, model_name, prompt):
        try:
            result = subprocess.run(['ollama', 'run', model_name, prompt], stdout=subprocess.PIPE, text=True, check=True)
            print(f"Model Output: {result.stdout}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running the model: {e}")

    def execute_pipeline(self, model_name='fine_tuned_model', model_output_dir='fine_tuned_gpt2_model',
                         prompt="Paraphrase this text"):
        # Fine-tune the model
        self.finetune_gpt(output_model_dir=model_output_dir)

        # Ensure the quantization structure is correct
        if self.ensure_quantization_file_structure(model_output_dir):
            self.quantize_model(model_output_dir)

        # Correct the model path to point to the quantized GGUF file
        model_path = os.path.join(model_output_dir, "fine_tuned_gpt2_model.q4_K_M.gguf").replace("\\", "/")

        # Create the Modelfile for Ollama
        self.create_modelfile(model_name, model_path)

        # Build the model in Ollama
        self.build_ollama_model(model_name)

        # Run inference with a sample prompt
        output = self.run_ollama_model(model_name, f"Provide a paraphrase for the following text: '{prompt}'")
        return output


# Run the pipeline
if __name__ == '__main__':
    hdt_finetune_gpt = HDTFineTuneGPT2(do_extraction=False)
    hdt_finetune_gpt.execute_pipeline()