import os
import shutil


def copy_files_to_txt(current_path):
    # Recursively walk through the directory structure
    for root, dirs, files in os.walk(current_path):
        # Iterate over all the files in the current directory
        for file in files:
            file_path = os.path.join(root, file)

            # Only process non-txt files (to avoid overwriting .txt files)
            if (file.endswith('.py') or file.endswith('.bat') or file.endswith('.ps') or file.endswith('.bat') or file.endswith('.cmd') or file.endswith('.requirements.txt'))\
                    and (not file in 'node_modules' and not file in 'dist' and not file in 'venv'):
                # Open the original file and read its content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Create the destination .txt file path
                txt_file_path = os.path.join(current_path,'self_analyse','code_base', file + '.txt')

                # Write the content to the new .txt file
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(content)

def merge_txt_files(current_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as merged_file:
        # Recursively walk through the directory structure
        for root, dirs, files in os.walk(current_path):
            for file in files:
                # Only process .txt files
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Write file content to the merged file
                        merged_file.write(f"--- Start of {file} ---\n")
                        merged_file.write(content)
                        merged_file.write(f"\n--- End of {file} ---\n\n")

# Run the function for the current directory




# Run the function for the current directory
current_dir = os.getcwd()
root_directory = os.path.join(current_dir, '..')
copy_files_to_txt(root_directory)

output_file = os.path.join(current_dir, 'merged_code_rag_flow.txt')
merge_txt_files(current_dir, output_file)
