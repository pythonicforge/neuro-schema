from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch


class LocalLLM:
    def __init__(self, model_id="TheBloke/NeuralBeagle14.5-4B"):
        print("Loading local model... (this may take a minute!)")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens=150, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
