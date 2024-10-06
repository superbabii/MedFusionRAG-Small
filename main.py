import os
import random
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import openai
from Bio import Entrez
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Set environment variable to allow duplicated OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load OpenAI API key from environment variables
openai.api_key = "sk-proj-u3PmINOij2w92y0cdl3xT3BlbkFJm3T5yhQttwfkkdp2rNdG"

# Replace your email for NCBI API
Entrez.email = "nzrbabii@gmail.com"

# Function to load random question and its correct answer
def load_random_question_with_answer(mmlu_path):
    try:
        with open(mmlu_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Randomly select a question
            random_question = random.choice(list(data.values()))
            question = random_question['question']
            options = random_question['options']
            correct_answer = random_question['answer']
            return question, options, correct_answer
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {mmlu_path}: {e}")
    except Exception as e:
        print(f"Error loading or parsing {mmlu_path}: {e}")
    return None, None, None

# Function to extract keywords using POS tagging and stopword removal
def extract_keywords(question):
    # Tokenize and remove stopwords
    words = word_tokenize(question)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

    # POS tagging to extract relevant words (nouns, adjectives)
    pos_tags = pos_tag(filtered_words)
    
    # Use only specific part-of-speech tags for extracting important terms
    important_keywords = [word for word, tag in pos_tags if tag.startswith('NN') or tag.startswith('JJ')]
    
    # Further filter keywords to remove generic terms
    important_keywords = [word for word in important_keywords if len(word) > 3]  # Remove very short words
    important_keywords = [word for word in important_keywords if not re.match(r'^[a-zA-Z]$', word)]  # Remove single letters
    important_keywords = list(set(important_keywords))  # Remove duplicates
    
    # Limit the number of keywords to avoid overly broad queries (choose top 5 most relevant)
    important_keywords = important_keywords[:10]
    
    # Join keywords with "AND" to form the final query
    query = " OR ".join(important_keywords)
    print(f"Extracted Keywords for PubMed Query: {query}")
    return query

# Function to fetch PubMed articles based on a query
def fetch_pubmed_articles(query, max_articles=10, retries=3, delay=5):
    articles = []
    for attempt in range(retries):
        try:
            print(f"Searching PubMed with query: {query}")  # Debugging: Print the exact query being sent
            search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles)
            search_results = Entrez.read(search_handle)
            search_handle.close()
            print(search_results)
            ids = search_results["IdList"]

            for pubmed_id in ids:
                fetch_handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="abstract", retmode="text")
                abstract = fetch_handle.read().strip()
                fetch_handle.close()
                if abstract:
                    articles.append({"id": pubmed_id, "text": abstract})
            return articles
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
    return []

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

# Generator class to interact with OpenAI API
class Generator:
    def __init__(self, model="gpt-4"):
        self.model = model
    
    def generate_answer(self, question, context, options):
        prompt = f"""
        You are a helpful medical expert. Given the following context from medical documents, answer the question.
        
        Context:
        {context}
        
        Question: {question}
        Options: {', '.join(options.values())}
        
        Select the best answer by choosing one of the options (A, B, C, etc.). Respond with only the letter of the option.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0
            )
            generated_text = response['choices'][0]['message']['content']
            print(f"Generated Text: {generated_text}")
            
            # Extract the option from the generated text
            for key, value in options.items():
                if value in generated_text:
                    return key
            # Fallback to the first option if no match is found
            return list(options.keys())[0]
        
        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {str(e)}")
            return list(options.keys())[0]

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
    for key, option in options.items():
        print(f"{key}. {option}")    
    print(f"Correct Answer: {correct_answer}")
    
    # Fetch PubMed articles based on the question and options
    print("Fetching PubMed articles...")
    query = extract_keywords(question)
    fetched_articles = fetch_pubmed_articles(query, max_articles=10)
    
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
        
        # Compare the selected answer with the correct answer
        is_correct = selected_answer.strip().upper() == correct_answer.strip().upper()
        print("Correct:", is_correct)
    else:
        print("Error: No valid documents for retrieval.")

# Run the main function
if __name__ == "__main__":
    main()