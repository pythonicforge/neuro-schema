from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch


class LocalLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        print("Loaing local model... (this may take a minute!)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(self, prompt: str, max_new_tokens=150,  temperature=0.7):
        outputs = self.pipeline(
            prompt,
            max_new_tokens = max_new_tokens,
            do_sample= True,
            temperature = temperature,
            top_k = 50,
            top_p = 0.95,
            num_return_sequences=1
        )
        return outputs[0]['generated_text']