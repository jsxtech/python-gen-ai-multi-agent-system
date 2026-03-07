from dotenv import load_dotenv
from agents import MultiAgentSystem

load_dotenv()

system = MultiAgentSystem()

# Add sample documents
system.rag_agent.add_documents([
    "Python is a high-level programming language.",
    "AI agents can perform autonomous tasks.",
    "RAG combines retrieval with generation."
])

# Query with RAG
response = system.process("What is RAG?")
print(f"Response: {response}")
