from llama_cpp import Llama
from scripts import logger

class QuantizedLLM:
    def __init__(self, model_path, context_length=2048, threads=8):
        try:
            logger.info("Preparing quantised model")
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_threads=threads,
                chat_format="chatml"
            )
            logger.success("Quantised model loaded succesfully")
        except Exception as e:
            logger.critical(e)

    def generate(self, prompt, max_tokens=256, temp=0.7, top_p=0.95):
        output = self.model(prompt, max_tokens=max_tokens, temperature=temp, top_p=top_p)
        return output["choices"][0]["text"]