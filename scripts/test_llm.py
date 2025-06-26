# from core.llm_wrapper import load_llm

# llm = load_llm()

# response = llm.generate("You are an AI assistant. Reply to: What is conciousness?")
# print(f"RESPONSE: {response}")

from core.llm_wrapper import TransformerLLM, QuantizedLLM

# transformed_model = TransformerLLM(model_id="cognitivecomputations/TinyDolphin-2.8-1.1b")
# transformed_model.quantize_to_gguf()

quantized_model = QuantizedLLM(model_path="models/quantized/TinyDolphin-2.8-1.1b-f16.gguf")
response = quantized_model.generate("Can you explain me what is conciousness?")
print(f"RESPONSE: {response}")