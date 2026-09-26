"""Unit tests for the multi-agent system.

These tests mock the OpenAI client and ChromaDB collection so they run
without a real API key or network access.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agents
from agents import ChatAgent, MultiAgentSystem, RAGAgent, _resolve_api_key


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------


def test_resolve_api_key_explicit():
    assert _resolve_api_key("sk-explicit") == "sk-explicit"


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert _resolve_api_key(None) == "sk-env"


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        _resolve_api_key(None)


# ---------------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------------


def _make_chat_agent(**kwargs) -> ChatAgent:
    with patch.object(agents, "OpenAI") as mock_openai:
        agent = ChatAgent(api_key="sk-test", **kwargs)
        agent._mock_client = mock_openai.return_value  # type: ignore[attr-defined]
    return agent


def _set_completion(agent: ChatAgent, content):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    agent.client.chat.completions.create = MagicMock(return_value=response)


def test_chat_agent_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ChatAgent(api_key=None)


def test_chat_agent_negative_max_history():
    with pytest.raises(ValueError):
        _make_chat_agent(max_history=-1)


def test_respond_rejects_empty_message():
    agent = _make_chat_agent()
    with pytest.raises(ValueError):
        agent.respond("   ")


def test_respond_returns_content_and_records_history():
    agent = _make_chat_agent()
    _set_completion(agent, "hello there")
    result = agent.respond("hi")
    assert result == "hello there"
    assert agent.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


def test_respond_none_content_raises_runtime_error():
    agent = _make_chat_agent()
    _set_completion(agent, None)
    with pytest.raises(RuntimeError, match="no content"):
        agent.respond("hi")
    # History must not be polluted by a failed turn
    assert agent.history == []


def test_respond_wraps_api_errors():
    agent = _make_chat_agent()
    agent.client.chat.completions.create = MagicMock(side_effect=Exception("boom"))
    with pytest.raises(RuntimeError, match="Chat completion failed"):
        agent.respond("hi")


def test_history_is_trimmed_to_max():
    agent = _make_chat_agent(max_history=4)
    _set_completion(agent, "ok")
    for _ in range(5):
        agent.respond("question")
    # 5 turns = 10 messages, capped to 4 (2 whole turns)
    assert len(agent.history) == 4
    assert agent.history[0]["role"] == "user"


def test_clear_history():
    agent = _make_chat_agent()
    _set_completion(agent, "ok")
    agent.respond("hi")
    agent.clear_history()
    assert agent.history == []


def test_context_injected_into_system_prompt():
    agent = _make_chat_agent()
    _set_completion(agent, "ok")
    agent.respond("hi", context="Relevant context: foo")
    sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert "Relevant context: foo" in sent[0]["content"]


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------


def _make_rag_agent() -> RAGAgent:
    with patch.object(agents, "chromadb") as mock_chroma, patch.object(
        agents, "embedding_functions"
    ):
        agent = RAGAgent(api_key="sk-test")
        agent.collection = MagicMock()
        _ = mock_chroma
    return agent


def test_rag_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        RAGAgent(api_key=None)


def test_add_documents_empty_list():
    agent = _make_rag_agent()
    with pytest.raises(ValueError):
        agent.add_documents([])


def test_add_documents_non_string_items():
    agent = _make_rag_agent()
    with pytest.raises(ValueError):
        agent.add_documents(["ok", "   "])


def test_add_documents_id_length_mismatch():
    agent = _make_rag_agent()
    with pytest.raises(ValueError):
        agent.add_documents(["a", "b"], ids=["only-one"])


def test_add_documents_generates_ids():
    agent = _make_rag_agent()
    agent.add_documents(["a", "b"])
    kwargs = agent.collection.upsert.call_args.kwargs
    assert len(kwargs["ids"]) == 2
    assert kwargs["documents"] == ["a", "b"]


def test_search_empty_query():
    agent = _make_rag_agent()
    with pytest.raises(ValueError):
        agent.search("  ")


def test_search_invalid_n():
    agent = _make_rag_agent()
    with pytest.raises(ValueError):
        agent.search("q", n=0)


def test_search_empty_collection_returns_empty():
    agent = _make_rag_agent()
    agent.collection.count.return_value = 0
    assert agent.search("q") == []


def test_search_caps_n_to_collection_size():
    agent = _make_rag_agent()
    agent.collection.count.return_value = 2
    agent.collection.query.return_value = {"documents": [["a", "b"]]}
    result = agent.search("q", n=10)
    assert result == ["a", "b"]
    assert agent.collection.query.call_args.kwargs["n_results"] == 2


# ---------------------------------------------------------------------------
# MultiAgentSystem
# ---------------------------------------------------------------------------


def test_process_without_rag_skips_search():
    with patch.object(agents, "OpenAI"), patch.object(agents, "chromadb"), patch.object(
        agents, "embedding_functions"
    ):
        system = MultiAgentSystem(api_key="sk-test")
    system.rag_agent.search = MagicMock()
    _set_completion(system.chat_agent, "answer")
    result = system.process("hi", use_rag=False)
    assert result == "answer"
    system.rag_agent.search.assert_not_called()


def test_process_with_rag_injects_context():
    with patch.object(agents, "OpenAI"), patch.object(agents, "chromadb"), patch.object(
        agents, "embedding_functions"
    ):
        system = MultiAgentSystem(api_key="sk-test")
    system.rag_agent.search = MagicMock(return_value=["doc1", "doc2"])
    _set_completion(system.chat_agent, "answer")
    result = system.process("hi")
    assert result == "answer"
    sent = system.chat_agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert "doc1 doc2" in sent[0]["content"]
