import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


# ── Shared helpers ────────────────────────────────────────────────────


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


# ── Search tool fixtures ─────────────────────────────────────────────


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


# ── Mock RAG system ──────────────────────────────────────────────────


@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem with sensible defaults."""
    rag = MagicMock()
    rag.query.return_value = ("Test answer", [])
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return rag


# ── Test FastAPI app & client ─────────────────────────────────────────
# Defines API endpoints inline to avoid importing backend/app.py,
# which mounts static files from a path that doesn't exist in tests.


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def _build_test_app(rag_system) -> FastAPI:
    """Build a minimal FastAPI app with the same endpoints as app.py."""
    test_app = FastAPI()

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return test_app


@pytest.fixture
def test_client(mock_rag_system):
    """FastAPI TestClient wired to a mock RAG system."""
    app = _build_test_app(mock_rag_system)
    return TestClient(app)
