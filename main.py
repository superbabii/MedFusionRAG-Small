import os

# Set environment variable to allow duplicated OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from retriever import Retriever
from generator import Generator

# Function to load the corpus
def load_statpearls_corpus():
    """
    Example of loading the StatPearls corpus from a JSON file or text files.
    Modify this according to your actual corpus format.
    """
    # For a JSON corpus
    import json
    with open('data/statpearls_corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    return [doc['text'] for doc in corpus]

# Function to load document based on ID
def load_document(doc_id):
    """
    Loads the document content based on the document ID (index).
    """
    corpus = load_statpearls_corpus()  # Load the StatPearls corpus
    return corpus[doc_id]  # Return the document text by its index


# Load retriever and generator
retriever = Retriever(index_path='D:/faiss/index')
generator = Generator()

# Example question
question = "What is the treatment for hypertension in elderly patients?"
options = ["A. Medication A", "B. Medication B", "C. Lifestyle change", "D. Surgery"]

# Retrieve relevant documents (indexes of top documents)
retrieved_docs = retriever.retrieve(question, top_k=5)

# Flatten retrieved_docs in case it's a 2D array (e.g., [[1, 2, 3]])
retrieved_docs = retrieved_docs.flatten()  # Ensure we have a flat array of document indices

# Load corresponding document content from pre-indexed snippets
docs_content = "\n".join([load_document(doc_id) for doc_id in retrieved_docs])

# Generate an answer
answer = generator.generate_answer(question, docs_content, options)

print("Generated Answer:", answer)
