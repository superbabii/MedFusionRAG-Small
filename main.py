#main.py
import os
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import openai
from Bio import Entrez

# Set environment variable to allow duplicated OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load OpenAI API key from environment variables
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "sk-proj-u3PmINOij2w92y0cdl3xT3BlbkFJm3T5yhQttwfkkdp2rNdG"

# Set your email for NCBI API
Entrez.email = "nzrbabii@gmail.com"

# Function to fetch PubMed articles based on a query
def fetch_pubmed_articles(query, max_articles=100):
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    
    ids = search_results["IdList"]
    articles = []
    
    for pubmed_id in ids:
        fetch_handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="abstract", retmode="text")
        abstract = fetch_handle.read()
        fetch_handle.close()
        
        # Store each article as a dictionary
        articles.append({"id": pubmed_id, "text": abstract.strip()})
    
    return articles

# Function to fetch articles based on a medical question and options
def fetch_articles_for_question_and_options(question, options, max_articles=10):
    combined_query = question + " " + " ".join(options)  # Combine the question and options into a single search query
    return fetch_pubmed_articles(combined_query, max_articles)

# Function to embed text using HuggingFace model
def embed_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to build and save FAISS index
def build_faiss_index(corpus, index_path):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    embeddings = []
    for doc in corpus:
        embeddings.append(embed_text(model, tokenizer, doc))
    
    embeddings = np.vstack(embeddings)  # Convert list of embeddings to a NumPy array
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)  # Add embeddings to the index
    
    # Ensure the directory exists
    output_dir = os.path.dirname(index_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index has been saved to {index_path}")

# Function to load FAISS index
def load_faiss_index(index_path):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        print(f"Error: FAISS index at {index_path} not found.")
        exit(1)

# Function to retrieve documents using FAISS
def retrieve_documents(index, query, model, tokenizer, corpus, top_k=5):
    query_embedding = embed_text(model, tokenizer, query)
    D, I = index.search(query_embedding, top_k)
    return [corpus[idx] for idx in I.flatten()]

# Generator class to interact with OpenAI API
class Generator:
    def __init__(self, model="gpt-4"):
        self.model = model
    
    def generate_answer(self, question, context, options):
        prompt = f"""
        You are a helpful medical expert. You are provided with the following context from medical documents:
        Context: {context}
        
        The question is: {question}
        The answer must be chosen from the following options: {options}
        
        If the context does not contain direct information, use your medical knowledge to provide an answer based on general medical understanding.
        Think step by step and then choose the best answer.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0
            )
            return response['choices'][0]['message']['content']
        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {str(e)}")
            return "Error generating answer. Please try again."

# Main function to combine the process
def main():
    # Paths
    index_path = 'D:/faiss/index'
    corpus_path = 'corpus.json'
    
    # Take dynamic input from user
    question = input("Enter your medical question: ")
    options = input("Enter the options separated by commas (e.g., 'A. Option1, B. Option2, C. Option3'): ").split(", ")
    
    # Fetch PubMed articles based on the question and options
    print("Fetching PubMed articles...")
    fetched_articles = fetch_articles_for_question_and_options(question, options, max_articles=10)
    
    # Save fetched articles to a corpus file (for future use)
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(fetched_articles, f, ensure_ascii=False, indent=4)
    print(f"Fetched articles have been saved to {corpus_path}")
    
    # Extract article texts
    corpus = [article['text'] for article in fetched_articles]
    
    # Build FAISS index
    print("Building FAISS index...")
    build_faiss_index(corpus, index_path)
    
    # Load FAISS index
    index = load_faiss_index(index_path)
    
    # Load tokenizer and model for embeddings
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize the generator
    generator = Generator()
    
    # Retrieve relevant documents (top_k = 5)
    retrieved_docs = retrieve_documents(index, question, model, tokenizer, corpus, top_k=5)
    
    # Combine the retrieved documents into a single context
    docs_content = "\n".join(retrieved_docs)
    
    # Generate an answer using GPT-4
    answer = generator.generate_answer(question, docs_content, options)
    
    # Output the generated answer
    print("Generated Answer:", answer)

# Run the main function
if __name__ == "__main__":
    main()