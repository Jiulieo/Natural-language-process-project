import torch
from transformers import pipeline

class LLMInterface:
    #di default carica il modello Qwen 1.8 bilion parameter
    def __init__(self, model_id = "Qwen/Qwen1.5-1.8B-Chat"):
        print(f"caricamente del modello {model_id}")

        #se possibile usa la gpu (cuda)
        self.device = 0 if torch.cuda.is_available else -1

        #con le gpu è conveniente usare float16 perchè è ottimizzato e va più veloce di float32(single precision) anche se più impreciso
        dtype = torch.float16 if self.device == 0 else torch.float32

        self.pipe = pipeline(
            "text-generation",
            model = model_id,
            device= self.device,
            torch_dtype = dtype
        )
        print("modello caricato")

    #send the prompt (logic problem) to the model and generate an answer
    def prompt_model(self, prompt, max_new_tokens = 150, temperature = 0.7):
        #il prompt viene formattato
        messages = [
            {"role":"system","content": "Sei un esperto di logica, risolvi il problema ragionando passo per passo"},
            {"role":"user","content":prompt}
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            do_sample = True if temperature>=0 else False
        )

        #Extract the answer
        answer = outputs[0]["generated-text"][-1]["content"]
        return answer.strip()

