# LangChain RAG System 🚀

A modular, production-ready Retrieval-Augmented Generation (RAG) REST API built with **FastAPI**, **LangChain**, and support for **Qdrant** and **ChromaDB**. 

Features multi-provider LLM & embedding backends (Ollama, LM Studio, OpenAI, Google Gemini), language-aware code and document chunking, hybrid retrieval (Dense + Sparse BM25 with Reciprocal Rank Fusion), and conversational question-answering with chat history reformulation.

---

## 🌟 Key Features

- **Multi-Vector DB Support**: Seamlessly switch between **Qdrant** (with native FastEmbed BM25 sparse vectors) and **ChromaDB** (with synchronized persistent BM25 hybrid ranking).
- **Hybrid Retrieval (Dense + BM25)**: Combines semantic dense embeddings with lexical BM25 matching fused using **Reciprocal Rank Fusion (RRF)** for optimal retrieval accuracy.
- **Pluggable Model Providers**: Configure any combination of LLMs and embedding models across:
  - **Ollama** (Local models like Gemma, Llama, Mistral)
  - **LM Studio** (Local OpenAI-compatible API)
  - **OpenAI** (GPT-4o, text-embedding-3, etc.)
  - **Google Gemini** (Gemini 1.5/2.0/3.0 series, Generative AI embeddings)
- **Universal Document & Code Loader**:
  - Documents: PDF (`PyPDFium2`), DOCX (`docx2txt`), PPTX (`unstructured`), Markdown, CSV, plain text.
  - Code & Notebooks: Jupyter Notebooks (`.ipynb`), Python, JavaScript, TypeScript, Java, C#, HTML, CSS, SQL, YAML, Dockerfile, and more.
  - Remote Repositories: Clone and ingest any public Git repository directly by URL.
- **Syntax-Aware Recursive Chunking**: Language-specific chunking for programming languages and markdown to preserve syntactic context, with deterministic chunk ID deduplication (`uuid5`).
- **Conversational RAG Chain**: Reformulates user queries based on conversation history using `RunnableBranch` and `RunnableParallel` before retrieving relevant context.
- **Production-Ready FastAPI**:
  - Interactive API docs via Swagger UI (`/api/docs`) and ReDoc (`/api/redoc`).
  - Concurrent health checks (`/api/health`) verifying API, LLM, embedding model, and vector DB connectivity.
  - Granular collection and document lifecycle management (create, list, inspect stats, delete by file).

---

## 🏗️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Client / Frontend UI   │
                          └─────────────┬────────────┘
                                        │ HTTP / JSON
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                FastAPI Server                                   │
│                                                                                 │
│   /api/health            /api/collections          /api/upload        /api/chat │
└─────────┬───────────────────────┬───────────────────────┬───────────────────┬───┘
          │                       │                       │                   │
          ▼                       ▼                       ▼                   ▼
┌──────────────────┐    ┌──────────────────┐    ┌───────────────────┐   ┌─────────────────┐
│  Health Checker  │    │ Collection Mgmt  │    │ Ingestion Engine  │   │  RAG Chain      │
│  (LLM, Embeds,   │    │ (Create, Delete, │    │ - SmartLoader     │   │ - Query Rewriter│
│   Vector DB)     │    │  Stats, List)    │    │ - Code Chunking   │   │ - Hybrid Search │
└──────────────────┘    └──────────────────┘    │ - UUID5 Dedupe    │   │ - LLM Generator │
                                                └─────────┬─────────┘   └────────┬────────┘
                                                          │                      │
                                                          ▼                      ▼
                                        ┌─────────────────────────────────────────────────┐
                                        │            Vector Storage & Retrieval           │
                                        │                                                 │
                                        │  • Qdrant (Dense + FastEmbed Sparse RRF)        │
                                        │  • ChromaDB (Dense + Local BM25 RRF)            │
                                        └─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Langchain_RAG/
├── config/
│   ├── __init__.py           # Exported config getters
│   ├── api_schema.py         # Pydantic request and response schemas
│   └── schema.py             # Settings, provider validation, LLM & embedding factories
├── core/
│   ├── __init__.py           # Client exports
│   ├── chain.py              # Contextual RAG chain & chat history formatter
│   ├── chunking.py           # Language-aware recursive character chunking
│   ├── loader.py             # SmartLoaderManager for multi-format & git loading
│   └── vectordb.py           # Unified vector store interface, hybrid search & BM25
├── routes/
│   ├── __init__.py           # Router exports
│   ├── collections.py        # Collection CRUD endpoints
│   └── ingestion.py          # File and Git repository ingestion endpoints
├── chroma_vector_db/         # Local Chroma database persistence directory
├── main.py                   # FastAPI application entry point & health check
├── pyproject.toml            # Project dependencies and packaging metadata
└── uv.lock                   # Deterministic dependency lockfile
```

---

## ⚙️ Configuration (.env)

Create a `.env` file in the project root to configure your providers and storage options:

```dotenv
# ==========================================
# Chat LLM Configuration
# Providers: ollama | lmstudio | openai | google
# ==========================================
chatllm_provider=lmstudio
chatllm_base_url=http://localhost:1234/v1/
chatllm_model=google/gemma-4-e4b
chatllm_request_timeout=120

# ==========================================
# Embedding Model Configuration
# Providers: ollama | lmstudio | openai | google
# ==========================================
embedding_provider=lmstudio
embedding_base_url=http://localhost:1234/v1/
embedding_model=text-embedding-embeddinggemma-300m
embedding_size=768
embedding_request_timeout=60

# ==========================================
# Vector Database Configuration
# Options: chroma | qdrant
# ==========================================
vector_db_provider=chroma
chroma_db_path=./chroma_vector_db/

# If using Qdrant:
# vector_db_provider=qdrant
# qdrant_url=http://localhost:6333
# qdrant_api_key=your_optional_api_key

# ==========================================
# Cloud API Keys (if using openai / google)
# ==========================================
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=AIza...
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: `>= 3.10` (Python 3.14 supported)
- [**uv**](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- A running LLM & embedding instance (e.g. [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), or cloud API key)
- (Optional) [Qdrant](https://qdrant.tech/) running locally via Docker if using Qdrant:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jatin-Gurwani/Langchain_RAG.git
   cd Langchain_RAG
   ```

2. **Install dependencies**:
   Using `uv`:
   ```bash
   uv sync
   ```
   Or using standard `pip`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

3. **Configure environment**:
   Copy or edit `.env` according to your local setup.

4. **Run the server**:
   Using `uv`:
   ```bash
   uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   Or directly with `uvicorn`:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access Interactive Docs**:
   - Swagger UI: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
   - ReDoc: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)

---

## 📖 API Endpoints Reference

### 1. Operational & System Health
- `GET /api/` - Root endpoint with API version.
- `GET /api/health` - Concurrently checks availability of API, LLM, Embedding Model, and Vector DB.
  ```json
  {
    "api_services": true,
    "chat_LLM_services": true,
    "embedding_Model_services": true,
    "vectordb_services": true
  }
  ```

### 2. Collection Management
- `GET /api/collections/` - List all collections.
- `GET /api/collections/{collection_name}` - Get metadata, active status, total chunks, and file counts.
- `POST /api/collections/` - Create a new collection.
  ```bash
  curl -X POST "http://localhost:8000/api/collections/" \
       -H "Content-Type: application/json" \
       -d '{"name": "research_papers"}'
  ```
- `DELETE /api/collections/` - Delete an existing collection.
  ```bash
  curl -X DELETE "http://localhost:8000/api/collections/" \
       -H "Content-Type: application/json" \
       -d '{"name": "research_papers"}'
  ```

### 3. Document Ingestion
- `GET /api/upload/{collection_name}` - Inspect ingested files and chunk distribution in a collection.
- `POST /api/upload/files/{collection_name}` - Upload one or more local files (multipart/form-data).
  ```bash
  curl -X POST "http://localhost:8000/api/upload/files/research_papers" \
       -F "files=@/path/to/document.pdf"
  ```
- `POST /api/upload/git` - Clone and ingest a Git repository into a collection.
  ```bash
  curl -X POST "http://localhost:8000/api/upload/git" \
       -H "Content-Type: application/json" \
       -d '{
         "collection_name": "research_papers",
         "git_url": "https://github.com/fastapi/fastapi.git"
       }'
  ```
- `DELETE /api/upload/delete` - Remove all chunks originating from a specific file.
  ```bash
  curl -X DELETE "http://localhost:8000/api/upload/delete" \
       -H "Content-Type: application/json" \
       -d '{
         "collection_name": "research_papers",
         "file_name": "document.pdf"
       }'
  ```

### 4. Conversational RAG Chat
- `POST /api/chat` - Ask questions with optional conversation history.
  ```bash
  curl -X POST "http://localhost:8000/api/chat" \
       -H "Content-Type: application/json" \
       -d '{
         "collection_name": "research_papers",
         "query": "What are the core conclusions of the paper?",
         "history": [
           ["user", "Hi!"],
           ["ai", "Hello! How can I help you today?"]
         ]
       }'
  ```
  **Response**:
  ```json
  {
    "airesponse": "Based on the provided documents, the core conclusions are...",
    "history": [
      ["user", "Hi!"],
      ["ai", "Hello! How can I help you today?"],
      ["user", "What are the core conclusions of the paper?"],
      ["ai", "Based on the provided documents, the core conclusions are..."]
    ]
  }
  ```

---

## 🛠️ Development & Testing

Run unit tests and test coverage using `pytest`:

```bash
uv run pytest
```

With coverage report:
```bash
uv run pytest --cov=core --cov=config --cov=routes
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
