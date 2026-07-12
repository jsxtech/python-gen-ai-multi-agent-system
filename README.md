# Python Gen AI Multi-Agent System

Minimal multi-agent system with RAG (Retrieval-Augmented Generation) and chat capabilities using OpenAI and ChromaDB.

## Features

- **ChatAgent**: GPT-4 powered conversational agent with history
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

# Uses OPENAI_API_KEY from .env by default
system = MultiAgentSystem()

# Or pass config explicitly
system = MultiAgentSystem(
    api_key="sk-...",
    model="gpt-4",
    collection_name="my_docs",
    persist_dir="./chroma_db",
)

# Add documents for RAG
system.rag_agent.add_documents([
    "Python is a high-level programming language.",
    "AI agents can perform autonomous tasks.",
])

# Query with RAG context (default)
response = system.process("What is Python?")

# Query without RAG
response = system.process("Hello!", use_rag=False)

# Clear conversation history
system.chat_agent.clear_history()
```

## Run

```bash
python main.py
```

## Architecture

- **ChatAgent**: Handles OpenAI API interactions with conversation memory
- **RAGAgent**: Manages document embeddings and similarity search via ChromaDB
- **MultiAgentSystem**: Combines retrieval and generation for enhanced responses

### Error Handling

All methods raise exceptions on failure:
- `ValueError` for invalid inputs
- `RuntimeError` for API/database errors

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `OPENAI_API_KEY` env var | OpenAI API key |
| `model` | `gpt-4` | Chat model to use |
| `collection_name` | `docs` | ChromaDB collection name |
| `persist_dir` | `./chroma_db` | ChromaDB storage path |

Vector database persists in the `./chroma_db` directory.
