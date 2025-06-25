from core.llm_wrapper import load_llm

llm = load_llm()

response = llm.generate("You are an AI assistant. Reply to: What is conciousness?")
print(f"RESPONSE: {response}")