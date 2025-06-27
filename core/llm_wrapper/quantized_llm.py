from llama_cpp import Llama
from scripts import logger

class QuantizedLLM:
    def __init__(
        self,
        model_path,
        context_length=4096,
        threads=8,
        gpu_layers=None,
        chat_format="chatml"
    ):
        """
        model_path: path to your GGUF model
        context_length: max tokens
        threads: CPU threads
        gpu_layers: int or None; None → put all layers on GPU
        chat_format: 'chatml' or other
        """
        try:
            logger.info("Preparing quantised model with GPU support")
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_threads=threads,
                n_gpu_layers=gpu_layers if gpu_layers is not None else -1,
                use_mlock=True,
                chat_format=chat_format
            )
            logger.success("Quantised model loaded successfully with GPU backend")
        except Exception as e:
            logger.critical(f"Failed to load quantised model: {e}")

    def generate(self, prompt, max_tokens=256, temp=0.7, top_p=0.95):
        """
        Generate text. Returns the string.
        """
        try:
            resp = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p
            )
            return resp["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

    @staticmethod
    def show_system_info():
        """
        Prints llama-cpp-python system info to verify CUDA/cuBLAS backend.
        """
        import llama_cpp
        info = llama_cpp.llama_print_system_info()
        logger.info(f"LLama system info:\n{info}")
        return info
