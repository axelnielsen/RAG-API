import ollama

class CustomLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": repetition_penalty,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
        )
        return response['response']
