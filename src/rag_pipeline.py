# src/rag_pipeline.py
from retriever import Retriever
from generator import Generator

class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer_question(self, query):
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query)
        
        # Concatenate retrieved documents for generation
        context = " ".join(retrieved_docs)
        
        # Generate answer based on the retrieved context
        answer = self.generator.generate(context, query)
        return answer
