import asyncio
import os
from pathlib import Path
import subprocess


class ModelFineTuner:
    def __init__(self, model_path, data_files, output_dir):
        self.model_path = model_path
        self.data_files = data_files
        self.output_dir = output_dir

    async def fine_tune(self):
        command = [
            "ollamactl",  # or replace with path to llama-3.1's ollamactl executable if needed
            "train",
            f"--model_path={self.model_path}",
            f"--data_files={self.data_files}",
            f"--output_dir={self.output_dir}",
            "--num_train_epochs=30",  # adjust as necessary
            "--per_device_train_batch_size=4",  # adjust as necessary
        ]

        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            print("Fine-tuning completed successfully.")
        else:
            print(f"Error during fine-tuning:\n{stderr}")


async def watch_directory(path):
    model_fine_tuner = ModelFineTuner(
        "local:///path/to/your/downloaded_llama3.1_model",  # replace with your actual paths
        "/path/to/training_data.jsonl",
        "/tmp/",
    )

    while True:
        for filename in os.listdir(path):
            if filename == "triggerfile":
                print("Trigger file detected, starting fine-tuning...")
                await model_fine_tuner.fine_tune()

                # Remove the trigger file after processing it to avoid re-triggering on subsequent runs
                (Path(path) / filename).unlink()

        await asyncio.sleep(1)  # Adjust as necessary


async def main():
    directory = "/path/to/your/watch_directory"

    print(f"Monitoring {directory} for trigger files...")

    while True:
        await asyncio.sleep(1)  # Adjust as necessary

        if "triggerfile" in os.listdir(directory):
            print("Trigger file detected, starting fine-tuning...")

            model_fine_tuner = ModelFineTuner(
                "local:///path/to/your/downloaded_llama3.1_model",  # replace with your actual paths
                "/path/to/training_data.jsonl",
                "/tmp/",
            )

            await model_fine_tuner.fine_tune()

            # Remove the trigger file after processing it to avoid re-triggering on subsequent runs
            (Path(directory) / "triggerfile").unlink()