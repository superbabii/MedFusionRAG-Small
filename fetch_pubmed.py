from Bio import Entrez
import json

# Set your email for NCBI API
Entrez.email = "nzrbabii@gmail.com"

# Fetch articles from PubMed
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

# Get PubMed articles related to hypertension and anemia
hypertension_articles = fetch_pubmed_articles("hypertension", max_articles=10)
anemia_articles = fetch_pubmed_articles("anemia", max_articles=10)

# Combine articles into a single list
corpus = hypertension_articles + anemia_articles

# Save as JSON
with open('data/statpearls_corpus.json', 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=4)

print("Saved PubMed articles to statpearls_corpus.json")
