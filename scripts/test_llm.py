from core.llm_wrapper import load_llm

llm = load_llm()

response = llm.generate("You are an AI assistant. Reply to: hey, good morning!")
print(f"RESPONSE: {response}")