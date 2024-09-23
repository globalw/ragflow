

import streamlit as st
import pdfplumber
from io import BytesIO
import os

from PyPDF2 import PdfReader


# Function to read text from the selected PDF file
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# Streamlit app layout and functionality
st.title("PDF Viewer with Context Menu")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Display the selected PDF in a viewer (you can use any method to display PDFs)
    pdf_reader = PdfReader(uploaded_file)
    pages = []
    for page in pdf_reader.pages:
        pages.append(page.extract_text())

    st.write("PDF Content:")
    for i, page in enumerate(pages):
        st.write(f"Page {i + 1}: {page}")

    # Context menu button (example using a dummy context menu)
    if st.button("Context Menu"):
        os.system('echo "PDF Viewer and Context Menu Example" | x-terminal-emulator')  # Dummy command for demonstration
        st.write("Opening PDF in default viewer...")
else:
    st.write("Please upload a PDF file.")