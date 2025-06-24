from core import LocalLLM

llm = LocalLLM()
output = llm.generate("How's your day going?")
print("Output: ", output)