import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

os.environ['USER_AGENT'] = 'myagent'

class OdysseyQA:
    def __init__(self, base_url, model="llama3", embedding_model="nomic-embed-text", chunk_size=500, chunk_overlap=20):
        """
        Initializes the OdysseyQA system.

        Parameters:
        - base_url: The base URL where Ollama is hosted (e.g., 'http://localhost:11434').
        - model: The name of the model to be used (default is 'llama3').
        - embedding_model: The name of the embedding model (default is 'nomic-embed-text').
        - chunk_size: The size of the chunks for text splitting (default is 500).
        - chunk_overlap: The overlap between text chunks for better context (default is 20).
        """
        self.base_url = base_url
        self.model = model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = Ollama(base_url=self.base_url, model=self.model)
        self.embedding = OllamaEmbeddings(base_url=self.base_url, model=self.embedding_model)
        self.vectorstore = None

    def load_document(self, url):
        """
        Loads the document from the specified URL.

        Parameters:
        - url: The URL of the document (e.g., the Project Gutenberg link to the Odyssey).
        """
        loader = WebBaseLoader(url)
        data = loader.load()
        return data

    def split_text(self, documents):
        """
        Splits the document into smaller chunks using RecursiveCharacterTextSplitter.

        Parameters:
        - documents: The loaded document from load_document method.

        Returns:
        - A list of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        all_splits = text_splitter.split_documents(documents)
        return all_splits

    def create_vector_store(self, documents):
        """
        Creates the vector store using Chroma and embeds the document chunks.

        Parameters:
        - documents: A list of document chunks from split_text method.
        """
        self.vectorstore = Chroma.from_documents(documents=documents, embedding=self.embedding)

    def ask_question(self, question):
        """
        Asks a question to the Odyssey text using the LLM and vector store.

        Parameters:
        - question: The question string to be asked.

        Returns:
        - The model's answer to the question.
        """
        if not self.vectorstore:
            raise ValueError("Vector store is not initialized. Run create_vector_store() first.")

        # Retrieve relevant documents based on the question
        docs = self.vectorstore.similarity_search(question)
        if len(docs) == 0:
            return "No relevant documents found."

        # Create a RetrievalQA chain to process the question and documents
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever())
        result = qa_chain.invoke({"query": question})

        return result['result']


# Example Usage
if __name__ == "__main__":
    # Base URL for Ollama, assuming it's running locally
    base_url = 'http://localhost:11434'

    # Create an instance of OdysseyQA
    odyssey_qa = OdysseyQA(base_url=base_url)

    # Load the document (Odyssey from Project Gutenberg)
    document_url = "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm"
    documents = odyssey_qa.load_document(document_url)

    # Split the document into smaller chunks
    split_docs = odyssey_qa.split_text(documents)

    # Create the vector store with embedded document chunks
    odyssey_qa.create_vector_store(split_docs)

    # Ask a question
    question = "Who is Neleus and who is in Neleus' family?"
    answer = odyssey_qa.ask_question(question)

    # Print the answer
    print(answer)
