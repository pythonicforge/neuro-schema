from core.llm_wrapper import TransformerLLM

llm = TransformerLLM(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm.quantize_to_gguf()