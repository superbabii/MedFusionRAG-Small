from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import os

# Define or load your corpus
def load_statpearls_corpus():
    return [
        "Hypertension is common in elderly patients due to vascular changes.",
        "ACE inhibitors are commonly used to treat hypertension in the elderly.",
        # Add more documents as needed
    ]
    
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load corpus (StatPearls or another dataset)
corpus = load_statpearls_corpus()  # You need to write a function to load your data

# Embed documents
embeddings = []
for doc in corpus:  # Each 'doc' is a string, not a dictionary
    inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True)  # Directly pass 'doc'
    outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

# Build FAISS index
embeddings = np.vstack(embeddings)  # Convert list of embeddings to a NumPy array
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance-based index
index.add(embeddings)  # Add embeddings to the index

# Ensure the directory exists
output_dir = 'D:/faiss'
os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn't exist

# Save the FAISS index to disk
faiss.write_index(index, 'D:/faiss/index')
