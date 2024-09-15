import re


def preprocess_ticket(ticket_text):
    # Normalize whitespace
    ticket_text = re.sub(r'\s+', ' ', ticket_text).strip()

    # Split into sections
    sections = re.split(r'=== (HDT-\d+\.pdf) ===', ticket_text)

    ticket_data = []
    for i in range(1, len(sections), 2):
        ticket_id = sections[i].strip()
        ticket_content = sections[i + 1].strip()

        # Extract different parts of the ticket
        description_match = re.search(r'Description of ' + re.escape(ticket_id) + r':(.*?)(?=Comments:)',
                                      ticket_content, re.DOTALL)
        comments_match = re.search(r'Comments:(.*)', ticket_content, re.DOTALL)

        description = description_match.group(1).strip() if description_match else ''
        comments = comments_match.group(1).strip() if comments_match else ''

        ticket_data.append({
            'ticket_id': ticket_id,
            'description': description,
            'comments': comments
        })

    return ticket_data


# Example usage: read text from a file
with open('merged_hdt_content.txt', 'r') as file:
    ticket_text = file.read()
processed_tickets = preprocess_ticket(ticket_text)

for ticket in processed_tickets:
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Description: {ticket['description']}")
    print(f"Comments: {ticket['comments']}")
    print()
