#main.py
import os
import random
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
        
        # Ensure the abstract is not empty or too short
        if abstract.strip():
            articles.append({"id": pubmed_id, "text": abstract.strip()})
        else:
            print(f"Warning: Article {pubmed_id} has no abstract.")
    
    if not articles:
        print("Error: No valid articles were fetched from PubMed.")
    
    return articles

# Function to build and save FAISS index with better article validation
def build_faiss_index(corpus, pubmed_index_path):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    embeddings = []
    for doc in corpus:
        if doc.strip():  # Ensure the document is not empty
            try:
                embedding = embed_text(model, tokenizer, doc)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error embedding document: {doc[:100]}... - Error: {e}")
    
    if len(embeddings) == 0:
        print("Error: No embeddings were generated. FAISS index cannot be built.")
        return
    
    embeddings = np.vstack(embeddings)  # Convert list of embeddings to a NumPy array
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)  # Add embeddings to the index
    
    # Ensure the directory exists
    output_dir = os.path.dirname(pubmed_index_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index to disk
    faiss.write_index(index, pubmed_index_path)
    print(f"FAISS index has been saved to {pubmed_index_path}")

# Function to fetch articles based on a medical question and options
def fetch_articles_for_question_and_options(question, options, max_articles=10):
    # Join only relevant keywords from the options
    combined_query = question + " " + " ".join([opt.split('.')[1].strip() for opt in options])
    return fetch_pubmed_articles(combined_query, max_articles)

# Function to embed text using HuggingFace model
def embed_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to load FAISS index
def load_faiss_index(pubmed_index_path):
    if os.path.exists(pubmed_index_path):
        return faiss.read_index(pubmed_index_path)
    else:
        print(f"Error: FAISS index at {pubmed_index_path} not found.")
        exit(1)

# Function to retrieve documents using FAISS
def retrieve_documents(index, query, model, tokenizer, corpus, top_k=5):
    query_embedding = embed_text(model, tokenizer, query)
    D, I = index.search(query_embedding, top_k)
    return [corpus[idx] for idx in I.flatten()]

# Function to load random question and its correct answer
def load_random_question_with_answer(mmlu_path):
    try:
        with open(mmlu_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Randomly select a question
            random_question = random.choice(data)
            
            question = random_question['question']
            options = random_question['options']
            correct_answer = random_question['answer']
            
            return question, options, correct_answer
    except Exception as e:
        print(f"Error loading or parsing {mmlu_path}: {str(e)}")
        return None, None, None
    
# Generator class to interact with OpenAI API
class Generator:
    def __init__(self, model="gpt-4"):
        self.model = model
    
    def generate_answer(self, question, context, options):
        prompt = f"""
        You are a helpful medical expert. You are provided with the following context from medical documents:
        Context: {context}
        
        The question is: {question}
        The answer must be chosen from the following options: {', '.join(options)}
        
        If the context does not contain direct information, use your medical knowledge to provide an answer based on general medical understanding.
        Think step by step and then choose the best answer by selecting one of the options.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0
            )
            generated_text = response['choices'][0]['message']['content']
            print(f"Generated Text: {generated_text}")  # Debugging step
            
            # Extract the option from the generated text
            for option in options:
                if option in generated_text:
                    return option
            # Fallback to the first option if no match is found
            return options[0]
        
        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {str(e)}")
            return options[0]


# Main function to combine the process
def main():
    # Paths
    pubmed_index_path = 'D:/faiss/pubmed_index'
    pubmed_corpus_path = 'data/pubmed_corpus.json'
    mmlu_path = 'data/mmlu_med.json'  # Path to the MMLU dataset
    
    # Load random question and options from the MMLU dataset
    print("Loading a random question from the MMLU dataset...")
    question, options, correct_answer = load_random_question_with_answer(mmlu_path)
    
    # Check if a question and options were successfully loaded
    if not question or not options or not correct_answer:
        print("Error: Could not load a question, options, or the correct answer. Exiting.")
        return
    
    # Print the question and options
    print(f"Question: {question}")
    print("Options:")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    
    print(f"Correct Answer: {correct_answer}")
    
    # # Take dynamic input from user
    # question = input("Enter your medical question: ")
    # options = input("Enter the options separated by commas (e.g., 'A. Option1, B. Option2, C. Option3'): ").split(", ")
    
    # Fetch PubMed articles based on the question and options
    print("Fetching PubMed articles...")
    fetched_articles = fetch_articles_for_question_and_options(question, options, max_articles=10)
    
    # Save fetched articles to a corpus file (for future use)
    with open(pubmed_corpus_path, 'w', encoding='utf-8') as f:
        json.dump(fetched_articles, f, ensure_ascii=False, indent=4)
    print(f"Fetched articles have been saved to {pubmed_corpus_path}")
    
    # Extract article texts
    corpus = [article['text'] for article in fetched_articles]
    
    # Check if the corpus is empty
    print(f"Number of articles fetched: {len(corpus)}")
    if len(corpus) > 0:
        print(f"Sample article content: {corpus[0][:500]}")  # Print the first 500 characters of the first article
    else:
        print("Error: No valid articles were fetched. Exiting.")
        return
    
    # Build FAISS index
    print("Building FAISS index...")
    build_faiss_index(corpus, pubmed_index_path)
    
    # Load FAISS index if it was successfully built
    if os.path.exists(pubmed_index_path):
        index = load_faiss_index(pubmed_index_path)
    else:
        print("Error: FAISS index was not built.")
        return
    
    # Load tokenizer and model for embeddings
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize the generator
    generator = Generator()
    
    # Retrieve relevant documents (top_k = 5)
    if len(corpus) > 0:
        retrieved_docs = retrieve_documents(index, question, model, tokenizer, corpus, top_k=5)
        
        # Combine the retrieved documents into a single context
        docs_content = "\n".join(retrieved_docs)
        
        # Generate an answer using GPT-4
        selected_answer = generator.generate_answer(question, docs_content, options)
        
        # Output the selected answer
        print("Selected Answer:", selected_answer)
        
        # Extract only the first letter of the selected answer and correct answer for comparison
        selected_option = selected_answer.split('.')[0].strip()  # Extract 'A', 'B', 'C', etc.
        correct_option = correct_answer.strip()
        
        # Compare the selected answer with the correct answer
        is_correct = selected_option.strip().lower() == correct_option.strip().lower()
        print("Correct:", is_correct)
    else:
        print("Error: No valid documents for retrieval.")

# Run the main function
if __name__ == "__main__":
    main()