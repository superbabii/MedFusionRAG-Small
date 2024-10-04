#main.py
import os
from retriever import Retriever
from generator import Generator

# Set environment variable to allow duplicated OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to load the corpus
def load_statpearls_corpus():
    """
    Example of loading the StatPearls corpus from a JSON file.
    Modify this according to your actual corpus format.
    """
    # For a JSON corpus
    import json
    try:
        with open('data/statpearls_corpus.json', 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        return [doc['text'] for doc in corpus]
    except FileNotFoundError:
        print("Error: Corpus file not found.")
        exit(1)

# Function to load document based on ID
def load_document(doc_id):
    """
    Loads the document content based on the document ID (index).
    """
    corpus = load_statpearls_corpus()  # Load the StatPearls corpus
    try:
        return corpus[doc_id]  # Return the document text by its index
    except IndexError:
        print(f"Error: Document with ID {doc_id} not found in corpus.")
        return ""

# Load retriever and generator
retriever = Retriever(index_path='D:/faiss/index')
generator = Generator()

# Take dynamic input from user
question = input("Enter your medical question: ")
options = input("Enter the options separated by commas (e.g., 'A. Option1, B. Option2, C. Option3'): ").split(", ")

# Retrieve relevant documents (indexes of top documents)
retrieved_docs = retriever.retrieve(question, top_k=5)

# Load corresponding document content from pre-indexed snippets
docs_content = "\n".join([load_document(int(doc_id)) for doc_id in retrieved_docs])

# Generate an answer
answer = generator.generate_answer(question, docs_content, options)

# Display the generated answer
print("Generated Answer:", answer)

