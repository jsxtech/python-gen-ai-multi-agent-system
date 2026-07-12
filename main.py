import logging

from agents import MultiAgentSystem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    system = MultiAgentSystem()

    # Add sample documents
    system.rag_agent.add_documents([
        "Python is a high-level programming language.",
        "AI agents can perform autonomous tasks.",
        "RAG combines retrieval with generation.",
    ])

    # Query with RAG
    response = system.process("What is RAG?")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
