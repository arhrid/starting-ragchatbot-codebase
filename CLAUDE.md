# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Uses ChromaDB for vector storage, Anthropic Claude for AI responses, and FastAPI for the backend with a vanilla HTML/JS/CSS frontend.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (starts FastAPI on port 8000)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Web interface: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Architecture

The system follows a RAG pipeline: **Document Ingestion → Vector Storage → Semantic Search → AI Response Generation**.

### Backend (`backend/`)

- **`app.py`** — FastAPI app with two endpoints: `POST /api/query` (process questions) and `GET /api/courses` (list courses). Serves the frontend as static files from `../frontend`. Loads documents from `../docs` on startup.
- **`rag_system.py`** — Main orchestrator. Wires together all components: document processing, vector store, AI generator, session manager, and search tools.
- **`document_processor.py`** — Parses course documents (expects structured format: Course Title/Link/Instructor headers, then `Lesson N:` markers). Chunks text with sentence-aware splitting and configurable overlap.
- **`vector_store.py`** — ChromaDB wrapper with two collections: `course_catalog` (course metadata for name resolution) and `course_content` (chunked content for semantic search). Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`).
- **`ai_generator.py`** — Anthropic Claude client with tool-use support. Handles the agentic loop: sends query → receives tool call → executes tool → sends results back for final answer.
- **`search_tools.py`** — Tool abstraction layer. `CourseSearchTool` implements the `Tool` ABC and provides `search_course_content` for Claude's tool calling. `ToolManager` registers tools and manages execution.
- **`session_manager.py`** — In-memory conversation history tracking per session (no persistence).
- **`config.py`** — Centralized configuration via `Config` dataclass. Key settings: chunk size (800), chunk overlap (100), max results (5), max history (2).
- **`models.py`** — Pydantic models: `Course`, `Lesson`, `CourseChunk`.

### Frontend (`frontend/`)

Vanilla HTML/JS/CSS chat interface. Communicates with the backend via fetch to `/api/query`.

### Document Format (`docs/`)

Course documents are `.txt` files with a specific format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]
```

## Key Design Decisions

- **Tool-based RAG**: The AI uses Claude's tool calling to decide when/how to search, rather than always retrieving context. The `search_course_content` tool supports filtering by course name and lesson number.
- **Course name resolution**: Fuzzy course matching uses vector search on the `course_catalog` collection rather than exact string matching.
- **No database persistence for sessions**: Conversation history is in-memory only and resets on server restart.
- **ChromaDB stored at `backend/chroma_db/`**: Persistent vector database, created on first run.

## Environment

- Python 3.13+, managed with `uv`
- Always use `uv` to run the server and manage dependencies. Never use `pip` directly.
- Requires `ANTHROPIC_API_KEY` in `.env` file (copy from `.env.example`)
- Default AI model: `claude-sonnet-4-20250514`
