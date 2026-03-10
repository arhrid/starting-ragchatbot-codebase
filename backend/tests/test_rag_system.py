"""Tests for RAGSystem.query() — all heavy components are mocked."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pydantic import BaseModel, ValidationError
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


def make_search_results(documents=None, metadata=None, distances=None, error=None):
    docs = documents or []
    meta = metadata or []
    dists = distances or [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=meta, distances=dists, error=error)


# ── Inline copy of QueryResponse to test Pydantic validation ────────
# Mirrors app.py so we can test without importing FastAPI app globals.

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str


# ── Helpers ──────────────────────────────────────────────────────────


def _build_rag_system():
    """Build a RAGSystem with all heavy deps mocked out."""
    with patch("rag_system.DocumentProcessor"), \
         patch("rag_system.VectorStore"), \
         patch("rag_system.AIGenerator") as MockAI, \
         patch("rag_system.SessionManager") as MockSession:

        mock_config = MagicMock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "/tmp/chroma"
        mock_config.EMBEDDING_MODEL = "test"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "fake"
        mock_config.ANTHROPIC_MODEL = "test"
        mock_config.MAX_HISTORY = 2

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Wire up mocks for easy access
        rag.ai_generator = MagicMock()
        rag.session_manager = MagicMock()
        rag.session_manager.get_conversation_history.return_value = None

        return rag


# ── Tests ────────────────────────────────────────────────────────────


class TestRAGSystemQuery:

    def test_returns_response_and_sources_tuple(self):
        """query() returns (str, list) tuple."""
        rag = _build_rag_system()
        rag.ai_generator.generate_response.return_value = "answer"

        response, sources = rag.query("hello")
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_wraps_query_in_prompt(self):
        """AI is called with 'Answer this question about course materials: ...'."""
        rag = _build_rag_system()
        rag.ai_generator.generate_response.return_value = "ok"

        rag.query("what is RAG?")

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "Answer this question about course materials: what is RAG?" in call_kwargs["query"]

    def test_passes_tools_and_tool_manager(self):
        """AI generator receives tools and tool_manager."""
        rag = _build_rag_system()
        rag.ai_generator.generate_response.return_value = "ok"

        rag.query("q")

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is rag.tool_manager

    def test_resets_sources_after_retrieval(self):
        """tool_manager.reset_sources() is called after get_last_sources()."""
        rag = _build_rag_system()
        rag.ai_generator.generate_response.return_value = "ok"
        rag.tool_manager = MagicMock()
        rag.tool_manager.get_tool_definitions.return_value = []
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("q")

        rag.tool_manager.reset_sources.assert_called_once()

    def test_updates_session_history(self):
        """add_exchange is called when session_id is provided."""
        rag = _build_rag_system()
        rag.ai_generator.generate_response.return_value = "answer"

        rag.query("q", session_id="s1")

        rag.session_manager.add_exchange.assert_called_once_with("s1", "q", "answer")

    def test_works_without_session_id(self):
        """No error when session_id is None."""
        rag = _build_rag_system()
        rag.ai_generator.generate_response.return_value = "ok"

        response, sources = rag.query("q")
        assert response == "ok"
        rag.session_manager.add_exchange.assert_not_called()

    def test_ai_exception_propagates(self):
        """No error handling in query() → AI exceptions propagate."""
        rag = _build_rag_system()
        rag.ai_generator.generate_response.side_effect = RuntimeError("AI failed")

        with pytest.raises(RuntimeError, match="AI failed"):
            rag.query("q")

    def test_sources_validate_against_query_response(self):
        """Dict sources from tools must validate against QueryResponse(sources: List[dict])."""
        dict_sources = [{"name": "Course - Lesson 1", "link": "https://example.com"}]
        resp = QueryResponse(answer="ok", sources=dict_sources, session_id="s1")
        assert resp.sources == dict_sources

    def test_string_sources_fail_query_response_validation(self):
        """If sources were List[str], QueryResponse(sources: List[dict]) would reject them.

        This test documents the critical bug: the original code had sources: List[str]
        which would reject the List[dict] produced by CourseSearchTool.
        """
        str_sources = ["Course - Lesson 1"]
        # List[str] should fail validation for List[dict]
        # Actually Pydantic v2 is lenient here — str is not a dict, so this should fail.
        with pytest.raises(ValidationError):
            QueryResponse(answer="ok", sources=str_sources, session_id="s1")

    def test_empty_sources_validate_ok(self):
        """Empty sources list validates fine."""
        resp = QueryResponse(answer="ok", sources=[], session_id="s1")
        assert resp.sources == []

    def test_integration_real_tool_manager_mocked_store(self):
        """Integration: real ToolManager + CourseSearchTool with mocked VectorStore."""
        rag = _build_rag_system()
        # Replace AI to simulate a tool call that populates sources
        mock_store = MagicMock()
        mock_store.search.return_value = make_search_results(
            documents=["content"],
            metadata=[{"course_title": "Intro", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/l1"

        from search_tools import CourseSearchTool, ToolManager
        tm = ToolManager()
        tool = CourseSearchTool(mock_store)
        tm.register_tool(tool)
        rag.tool_manager = tm

        # Simulate what happens after AI calls the tool
        tool.execute(query="q")
        sources = tm.get_last_sources()
        tm.reset_sources()

        assert len(sources) == 1
        assert sources[0]["name"] == "Intro - Lesson 1"
        assert sources[0]["link"] == "https://example.com/l1"

        # Validate these sources work with QueryResponse
        resp = QueryResponse(answer="ok", sources=sources, session_id="s1")
        assert resp.sources[0]["name"] == "Intro - Lesson 1"
