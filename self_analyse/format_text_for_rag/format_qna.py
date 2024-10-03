import csv


def generate_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    user_text = '\n'.join(lines[::2])  # Every other line starting from index 0 (i.e., all lines under "USER")
    assistant_text = '\n'.join(lines[1::2])  # Every other line starting from index 1 (i.e., all lines under "ASSISTANT")

    data = [[user_text, assistant_text]]  # Create a list of lists with the user and assistant texts as the rows

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the CSV file without headers
        for row in data:
            writer.writerow(row)


# Specify the input and output files
input_file = 'Needed.txt'
output_file = 'question_answer_list.csv'

generate_csv(input_file, output_file)