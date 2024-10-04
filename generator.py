#generator.py
import openai
import os

# Load OpenAI API key from environment variables
# openai.api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = "sk-proj-u3PmINOij2w92y0cdl3xT3BlbkFJm3T5yhQttwfkkdp2rNdG"

class Generator:
    def __init__(self, model="gpt-4"):
        self.model = model
    
    def generate_answer(self, question, context, options):
        prompt = f"""
        You are a helpful medical expert. You are provided with relevant documents to answer a question.
        Context: {context}
        Question: {question}
        Options: {options}
        
        Prioritize using the context provided to answer the question, and if the context is incomplete, explain why.
        Then, think step by step and choose the best answer based on the options given.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
        return response['choices'][0]['message']['content']

