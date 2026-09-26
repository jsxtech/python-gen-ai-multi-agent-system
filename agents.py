import logging
import os
import uuid
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Resolve the OpenAI API key from the argument or environment."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("OPENAI_API_KEY must be set or passed explicitly")
    return resolved_key


class ChatAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        system_prompt: str = "You are a helpful assistant.",
        max_history: int = 20,
    ) -> None:
        if max_history < 0:
            raise ValueError("max_history must be non-negative")
        self.client = OpenAI(api_key=_resolve_api_key(api_key))
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: list[dict[str, str]] = []

    def respond(self, message: str, context: str = "") -> str:
        if not message or not message.strip():
            raise ValueError("message must be a non-empty string")

        system_content = self.system_prompt
        if context:
            system_content += f"\n\n{context}"

        messages = [{"role": "system", "content": system_content}] + self.history
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except Exception as e:
            logger.exception("OpenAI API call failed")
            raise RuntimeError(f"Chat completion failed: {e}") from e

        assistant_message = response.choices[0].message.content
        if assistant_message is None:
            raise RuntimeError("Chat completion returned no content")

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": assistant_message})
        self._trim_history()
        return assistant_message

    def _trim_history(self) -> None:
        """Keep at most `max_history` messages, preserving whole user/assistant turns."""
        if self.max_history and len(self.history) > self.max_history:
            # max_history counts individual messages; trim oldest, keep even count
            excess = len(self.history) - self.max_history
            # Round up to an even number so we drop complete turns
            excess += excess % 2
            self.history = self.history[excess:]

    def clear_history(self) -> None:
        self.history.clear()


class RAGAgent:
    def __init__(
        self,
        collection_name: str = "docs",
        persist_dir: str = "./chroma_db",
        api_key: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY must be set or passed explicitly")
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(api_key=resolved_key)
        self.collection = self.chroma.get_or_create_collection(
            collection_name, embedding_function=self.ef
        )

    def add_documents(self, texts: list[str], ids: Optional[list[str]] = None) -> None:
        if not texts:
            raise ValueError("texts must be a non-empty list")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("All items in texts must be non-empty strings")
        if ids and len(ids) != len(texts):
            raise ValueError("ids length must match texts length")

        try:
            if not ids:
                ids = [str(uuid.uuid4()) for _ in texts]
            self.collection.upsert(documents=texts, ids=ids)
            logger.info("Added %d documents to collection", len(texts))
        except Exception as e:
            logger.exception("Failed to add documents")
            raise RuntimeError(f"Failed to add documents: {e}") from e

    def search(self, query: str, n: int = 3) -> list[str]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if n < 1:
            raise ValueError("n must be at least 1")

        count = self.collection.count()
        if count == 0:
            return []

        # Cap n_results to collection size to avoid ChromaDB error
        n = min(n, count)

        try:
            results = self.collection.query(query_texts=[query], n_results=n)
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.exception("Search failed for query: %s", query)
            raise RuntimeError(f"Search failed: {e}") from e


class MultiAgentSystem:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        collection_name: str = "docs",
        persist_dir: str = "./chroma_db",
    ) -> None:
        self.chat_agent = ChatAgent(api_key=api_key, model=model)
        self.rag_agent = RAGAgent(
            collection_name=collection_name,
            persist_dir=persist_dir,
            api_key=api_key,
        )

    def process(self, query: str, use_rag: bool = True) -> str:
        context = ""
        if use_rag:
            docs = self.rag_agent.search(query)
            context = f"Relevant context: {' '.join(docs)}" if docs else ""

        return self.chat_agent.respond(query, context)
