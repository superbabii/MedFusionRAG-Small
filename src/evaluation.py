# src/evaluation.py
from sklearn.metrics import accuracy_score

class Evaluator:
    def __init__(self, rag_pipeline, dataset):
        self.rag_pipeline = rag_pipeline
        self.dataset = dataset

    def evaluate(self):
        predictions = []
        true_answers = []
        for entry in self.dataset:
            query, true_answer = entry['question'], entry['answer']
            predicted_answer = self.rag_pipeline.answer_question(query)
            predictions.append(predicted_answer)
            true_answers.append(true_answer)
        
        accuracy = accuracy_score(true_answers, predictions)
        return accuracy
