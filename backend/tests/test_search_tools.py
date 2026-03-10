"""Tests for CourseSearchTool and ToolManager."""

import pytest
from unittest.mock import MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_search_results(documents=None, metadata=None, distances=None, error=None):
    docs = documents or []
    meta = metadata or []
    dists = distances or [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=meta, distances=dists, error=error)


# ── CourseSearchTool.execute ─────────────────────────────────────────


class TestCourseSearchToolExecute:

    def test_success_returns_formatted_string(self, search_tool, mock_vector_store):
        """execute() returns formatted content and populates last_sources as List[dict]."""
        mock_vector_store.search.return_value = make_search_results(
            documents=["chunk text"],
            metadata=[{"course_title": "Intro", "lesson_number": 1}],
        )
        result = search_tool.execute(query="hello")

        assert "chunk text" in result
        assert "[Intro - Lesson 1]" in result

    def test_last_sources_are_dicts(self, search_tool, mock_vector_store):
        """last_sources items must be dicts with 'name' and 'link' keys."""
        mock_vector_store.search.return_value = make_search_results(
            documents=["text"],
            metadata=[{"course_title": "Intro", "lesson_number": 1}],
        )
        search_tool.execute(query="q")

        assert len(search_tool.last_sources) == 1
        src = search_tool.last_sources[0]
        assert isinstance(src, dict)
        assert "name" in src
        assert "link" in src

    def test_error_from_store(self, search_tool, mock_vector_store):
        """When store returns an error, execute() returns that error string."""
        mock_vector_store.search.return_value = make_search_results(error="db down")
        result = search_tool.execute(query="q")
        assert result == "db down"

    def test_empty_results(self, search_tool, mock_vector_store):
        """Empty (non-error) results → 'No relevant content found'."""
        mock_vector_store.search.return_value = make_search_results()
        result = search_tool.execute(query="q")
        assert "No relevant content found" in result

    def test_parameter_forwarding(self, search_tool, mock_vector_store):
        """course_name and lesson_number are forwarded to store.search()."""
        mock_vector_store.search.return_value = make_search_results()
        search_tool.execute(query="q", course_name="MCP", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="q", course_name="MCP", lesson_number=3
        )

    def test_store_raises_exception(self, search_tool, mock_vector_store):
        """If store.search() raises, execute() has no try/except → exception propagates."""
        mock_vector_store.search.side_effect = RuntimeError("connection lost")

        with pytest.raises(RuntimeError, match="connection lost"):
            search_tool.execute(query="q")


# ── ToolManager ──────────────────────────────────────────────────────


class TestToolManager:

    def test_execute_tool_delegates(self, tool_manager, mock_vector_store):
        """execute_tool returns string from the underlying tool."""
        mock_vector_store.search.return_value = make_search_results()
        result = tool_manager.execute_tool("search_course_content", query="q")
        assert isinstance(result, str)

    def test_get_last_sources_returns_dicts(self, tool_manager, mock_vector_store):
        """get_last_sources returns dict-based sources after a search."""
        mock_vector_store.search.return_value = make_search_results(
            documents=["doc"],
            metadata=[{"course_title": "C", "lesson_number": 1}],
        )
        tool_manager.execute_tool("search_course_content", query="q")
        sources = tool_manager.get_last_sources()

        assert len(sources) == 1
        assert isinstance(sources[0], dict)

    def test_reset_sources(self, tool_manager, mock_vector_store):
        """reset_sources clears last_sources on all tools."""
        mock_vector_store.search.return_value = make_search_results(
            documents=["doc"],
            metadata=[{"course_title": "C", "lesson_number": 1}],
        )
        tool_manager.execute_tool("search_course_content", query="q")
        tool_manager.reset_sources()

        assert tool_manager.get_last_sources() == []

    def test_unknown_tool(self, tool_manager):
        """Calling an unregistered tool returns an error string (not an exception)."""
        result = tool_manager.execute_tool("nonexistent", query="q")
        assert "not found" in result
