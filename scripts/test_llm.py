from core import LocalLLM

llm = LocalLLM()
output = llm.generate("Explain what is conciousness")
print("Output: ", output)