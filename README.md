# Python Gen AI Multi-Agent System

Minimal multi-agent system with RAG (Retrieval-Augmented Generation) and chat capabilities using OpenAI and ChromaDB.

## Features

- **ChatAgent**: GPT-4 powered conversational agent
- **RAGAgent**: Vector-based document retrieval with ChromaDB
- **MultiAgentSystem**: Orchestrates agents for context-aware responses

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Usage

```python
from agents import MultiAgentSystem

system = MultiAgentSystem()

# Add documents for RAG
system.rag_agent.add_documents([
    "Python is a high-level programming language.",
    "AI agents can perform autonomous tasks."
])

# Query with RAG context (default)
response = system.process("What is Python?")

# Query without RAG
response = system.process("Hello!", use_rag=False)
```

## Run

```bash
python main.py
```

## Architecture

- **ChatAgent**: Handles OpenAI API interactions
- **RAGAgent**: Manages document embeddings and similarity search
- **MultiAgentSystem**: Combines retrieval and generation for enhanced responses

Vector database persists in `./chroma_db` directory.
