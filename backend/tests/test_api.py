"""Tests for FastAPI endpoints — RAG system is fully mocked."""

import pytest


class TestQueryEndpoint:

    def test_successful_query(self, test_client, mock_rag_system):
        """POST /api/query returns answer, sources, and session_id."""
        mock_rag_system.query.return_value = (
            "Python is great",
            [{"name": "Intro - Lesson 1", "link": "https://example.com"}],
        )

        resp = test_client.post("/api/query", json={"query": "What is Python?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Python is great"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["name"] == "Intro - Lesson 1"
        assert data["session_id"] == "test-session-id"

    def test_query_with_session_id(self, test_client, mock_rag_system):
        """Provided session_id is forwarded and returned."""
        resp = test_client.post(
            "/api/query", json={"query": "hi", "session_id": "my-session"}
        )

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my-session"
        mock_rag_system.query.assert_called_once_with("hi", "my-session")

    def test_query_creates_session_when_missing(self, test_client, mock_rag_system):
        """When no session_id is sent, a new one is created."""
        resp = test_client.post("/api/query", json={"query": "hi"})

        assert resp.status_code == 200
        mock_rag_system.session_manager.create_session.assert_called_once()
        assert resp.json()["session_id"] == "test-session-id"

    def test_query_missing_body(self, test_client):
        """Missing request body returns 422."""
        resp = test_client.post("/api/query")
        assert resp.status_code == 422

    def test_query_missing_query_field(self, test_client):
        """Body without 'query' field returns 422."""
        resp = test_client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_query_rag_error_returns_500(self, test_client, mock_rag_system):
        """RAG system exception is converted to 500."""
        mock_rag_system.query.side_effect = RuntimeError("AI exploded")

        resp = test_client.post("/api/query", json={"query": "boom"})

        assert resp.status_code == 500
        assert "AI exploded" in resp.json()["detail"]

    def test_query_empty_sources(self, test_client, mock_rag_system):
        """Empty sources list is valid."""
        mock_rag_system.query.return_value = ("No sources needed", [])

        resp = test_client.post("/api/query", json={"query": "general question"})

        assert resp.status_code == 200
        assert resp.json()["sources"] == []


class TestCoursesEndpoint:

    def test_successful_courses(self, test_client):
        """GET /api/courses returns course stats."""
        resp = test_client.get("/api/courses")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course A", "Course B"]

    def test_courses_empty(self, test_client, mock_rag_system):
        """Empty catalog returns zero courses."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        resp = test_client.get("/api/courses")

        assert resp.status_code == 200
        assert resp.json()["total_courses"] == 0
        assert resp.json()["course_titles"] == []

    def test_courses_error_returns_500(self, test_client, mock_rag_system):
        """Analytics exception is converted to 500."""
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db down")

        resp = test_client.get("/api/courses")

        assert resp.status_code == 500
        assert "db down" in resp.json()["detail"]


class TestRootEndpoint:

    def test_root_returns_404(self, test_client):
        """Test app has no static mount, so / returns 404."""
        resp = test_client.get("/")
        assert resp.status_code == 404
