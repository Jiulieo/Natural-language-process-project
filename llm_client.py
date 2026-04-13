import torch
from transformers import pipeline

class LLMInterface:
    #di default carica il modello Qwen 1.8 bilion parameter
    def __init__(self, model_id = "Qwen/Qwen1.5-1.8B-Chat"):
        print(f"caricamente del modello {model_id}")

        #se possibile usa la gpu (cuda)
        self.device = 0 if torch.cuda.is_available() else -1

        #float 16 is optimized for GPU
        dtype = torch.float16 if self.device == 0 else torch.float32

        self.pipe = pipeline(
            "text-generation",
            model = model_id,
            device= self.device,
            dtype = dtype
        )

        if hasattr(self.pipe.model.generation_config, "max_length"):
            self.pipe.model.generation_config.max_length = None
            
        print("modello caricato")

    #send the prompt (logic problem) to the model and generate an answer
    def prompt_model(self, prompt, max_new_tokens = 150, temperature = 0.7):
        messages = [
            {"role":"system","content": "You are an expert of logic problem, solve the following problem step by step:"},
            {"role":"user","content":prompt}
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens = max_new_tokens,
            max_length = None,
            temperature = temperature,
            do_sample = True if temperature>=0 else False,
            pad_token_id=self.pipe.tokenizer.eos_token_id #Useful for Qwen
        )

        #Extract the answer
        answer = outputs[0]["generated_text"][-1]["content"]
        return answer.strip()

