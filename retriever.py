from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

class Retriever:
    def __init__(self, index_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.index = faiss.read_index(index_path)
    
    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def retrieve(self, query, top_k=5):
        query_embedding = self.embed_text(query)
        D, I = self.index.search(query_embedding, top_k)
        return I  # Return top_k indices
