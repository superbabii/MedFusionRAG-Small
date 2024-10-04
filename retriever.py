#retriever.py
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

class Retriever:
    def __init__(self, index_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        # Load the tokenizer and model for sentence embedding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Load FAISS index
        self.index = faiss.read_index(index_path)
    
    def embed_text(self, text):
        # Tokenize and embed the query text
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Get the mean of the last hidden states for embedding
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def retrieve(self, query, top_k=10):
        # Embed the query
        query_embedding = self.embed_text(query)
        # Perform the FAISS search
        D, I = self.index.search(query_embedding, top_k)
        return I.flatten()  # Return a flat array of document indices
