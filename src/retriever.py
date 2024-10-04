# src/retriever.py
from elasticsearch import Elasticsearch

class Retriever:
    def __init__(self, index_name="statpearls"):
        self.es = Elasticsearch()
        self.index_name = index_name
    
    def retrieve(self, query, top_k=5):
        body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }
        response = self.es.search(index=self.index_name, body=body)
        return [hit["_source"]["text"] for hit in response["hits"]["hits"]]

