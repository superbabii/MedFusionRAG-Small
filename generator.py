import openai

openai.api_key = "sk-proj-u3PmINOij2w92y0cdl3xT3BlbkFJm3T5yhQttwfkkdp2rNdG"

class Generator:
    def __init__(self, model="gpt-4"):
        self.model = model
    
    def generate_answer(self, question, context, options):
        prompt = f"""
        You are a helpful medical expert. Answer the following question based on the relevant documents:
        Context: {context}
        Question: {question}
        Options: {options}
        
        Think step by step and then choose the best answer.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
        return response['choices'][0]['message']['content']
