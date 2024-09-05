import os
import streamlit as st
import random
import string
import json
# Define the iframe content
iframe_code = '''
<iframe
src="http://localhost:9222/chat/share?shared_id=ragflow-RkN2UyMmM0Njg5MjExZWY5N2Q5ZDhjYj"
style="width: 100%; height: 100%; min-height: 600px"
frameborder="0"
>
</iframe>
'''

# Set up the Streamlit page configuration
st.set_page_config(page_title="iFrame with File Upload", layout="wide")

# Render the iframe in the Streamlit app
st.markdown(iframe_code, unsafe_allow_html=True)

# Add a drag-and-drop file uploader
uploaded_file = st.file_uploader("Upload a file", type=None)

# If a file is uploaded
if uploaded_file is not None:
    # Define the path where the file will be saved
    save_path = os.path.join(os.getcwd(), uploaded_file.name)

    # Save the file to the local folder
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File saved: {save_path}")

# A function to generate a random name of length 10
def generate_random_name(length=10):
    return ''.join(random.choices(string.ascii_letters, k=length))

# Initialize session state to store the list of buttons and chat history
if 'buttons' not in st.session_state:
    st.session_state.buttons = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ""

# Layout for the app
left_col, right_col = st.columns([2, 1])

# Add the chat box on the left
with left_col:
    chat_input = st.text_area("Chat", value=st.session_state.chat_history, height=200)
    if st.button("Send"):
        st.session_state.chat_history += chat_input + "\n"

# Right column for buttons
with right_col:
    st.write("Dynamic Buttons")

    # Button to add a random name button to the list
    if st.button("Add Random Button"):
        random_name = generate_random_name()
        st.session_state.buttons.append(random_name)

    # Display all the dynamically created buttons
    for button_name in st.session_state.buttons:
        if st.button(button_name):
            st.write(f"You clicked {button_name}")

    # Button to parse chat to JSON
    if st.button("Parse Chat to JSON"):
        chat_lines = st.session_state.chat_history.strip().split("\n")
        chat_json = {"chat_history": chat_lines}
        st.json(chat_json)
