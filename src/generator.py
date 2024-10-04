# src/generator.py
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Generator:
    def __init__(self, model_name='t5-large'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, context, query, max_length=100):
        input_text = f"question: {query} context: {context} </s>"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

