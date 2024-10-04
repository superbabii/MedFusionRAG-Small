# run_pipeline.py
import json
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.generator import Generator
from src.evaluation import Evaluator

# Initialize your components
retriever = Retriever(index_name="statpearls", use_rrf=True)  # Custom retriever (RRF-4)
generator = Generator(model_name="t5-large")  # Fine-tuned generator

# Build the RAG pipeline
rag_pipeline = RAGPipeline(retriever, generator)

# Evaluate on MMLU-Med dataset
evaluator = Evaluator(rag_pipeline, dataset="data/MMLU-Med/test.json")
accuracy = evaluator.evaluate()

print(f"Accuracy of the RAG pipeline: {accuracy}")
