import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


def make_search_results(
    documents: List[str] = None,
    metadata: List[Dict[str, Any]] = None,
    distances: List[float] = None,
    error: Optional[str] = None,
) -> SearchResults:
    """Factory for SearchResults with sensible defaults."""
    docs = documents or []
    meta = metadata or []
    dists = distances or [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=meta, distances=dists, error=error)


@pytest.fixture
def mock_vector_store():
    """A MagicMock standing in for VectorStore."""
    store = MagicMock()
    store.search.return_value = make_search_results()
    store.get_lesson_link.return_value = "https://example.com/lesson"
    return store


@pytest.fixture
def search_tool(mock_vector_store):
    """CourseSearchTool wired to a mock store."""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(search_tool):
    """ToolManager with a registered CourseSearchTool."""
    tm = ToolManager()
    tm.register_tool(search_tool)
    return tm
