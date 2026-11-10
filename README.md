# LangChain RAG System 🚀

A modular, production-ready Retrieval-Augmented Generation (RAG) REST API built with **FastAPI**, **LangChain**, and support for **Qdrant** and **ChromaDB**.

Features multi-provider LLM & embedding backends (Ollama, Private/LM Studio, OpenAI, Google Gemini), universal document and language-aware code chunking, hybrid retrieval (Dense + Sparse BM25 fused with Reciprocal Rank Fusion), conversation history reformulation, and out-of-the-box Docker Compose deployment.

---

## 🌟 Key Features

- **Multi-Vector DB Support**:
  - **Qdrant**: Dense vectors + native FastEmbed BM25 sparse vectors with server-side Reciprocal Rank Fusion (`Fusion.RRF`).
  - **ChromaDB**: Persistent dense vectors paired with a synchronized local BM25Okapi index and client-side RRF ranking.
- **Hybrid Retrieval (Dense + BM25 with RRF)**:
  - Merges semantic dense similarity with exact keyword lexical matching using configurable Reciprocal Rank Fusion (`rrf_k`, `hybrid_prefetch_limit`).
  - Metadata filtering allowed on `file_name` and `file_type`.
- **Pluggable LLM & Embedding Backends**:
  - **`ollama`**: Local open-weights models (Gemma, Llama, Mistral, Qwen, etc.).
  - **`private`**: OpenAI-compatible local/private servers (LM Studio, vLLM, LocalAI, Ollama OpenAI endpoints).
  - **`openai`**: GPT-4o, GPT-4o-mini, text-embedding-3 series.
  - **`google`**: Gemini 1.5, 2.0, 3.x series and Generative AI embeddings.
- **Reva Conversational Assistant**:
  - Contextual query rewriting using `RunnableBranch` and `RunnableParallel` to resolve co-references against chat history.
  - Grounded answering with strict adherence to ingested context.
  - Formatted citations: Appends `(Filename: <name>, Page: <num>)` without leaking internal IDs or paths.
  - Code-first responses: Emits clean, comment-free code blocks when code snippets are requested.
- **Universal Document, Code & Git Loader**:
  - **Documents**: PDF (`PyPDFium2` / `PyMuPDF`), DOCX (`docx2txt`), PowerPoint (`.pptx`, `.ppt` via `unstructured`), Markdown (`.md`, `.mdx`), CSV, plain text (`.txt`, `.rst`, `.log`).
  - **Notebooks**: Jupyter Notebooks (`.ipynb`).
  - **Code & Configs**: Python, JavaScript, TypeScript, Java, C#, C, C++, HTML, CSS, SQL, JSON, YAML, TOML, XML, Bash, Dockerfile, and more.
  - **Git Repositories**: Ingest any public Git repository directly by URL with automatic filtering of binaries and non-text files.
- **Syntax-Aware Recursive Chunking & Deduplication**:
  - Language-specialized chunking for Python, JavaScript, TypeScript, Java, C#, HTML, and Markdown to preserve syntactic boundaries.
  - Deterministic chunk ID hashing using `uuid5(NAMESPACE_URL, file_name|page_content)` to prevent duplicates.
- **Production-Ready FastAPI & Docker**:
  - Interactive API documentation via Swagger UI (`/api/docs`) and ReDoc (`/api/redoc`).
  - Subsystem health checks (`/api/health`) verifying API, LLM, embedding model, and vector DB connectivity.
  - One-command containerized stack with Docker Compose (`compose.yaml`).

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
│  Health Checker  │    │ Collection Mgmt  │    │ Ingestion Engine  │   │  Reva Assistant │
│  (Chat LLM,      │    │ (Create, Delete, │    │ - SmartLoader     │   │ - Query Rewriter│
│   Embeddings,    │    │  Stats, List)    │    │ - Code Chunking   │   │ - Hybrid RRF    │
│   Vector DB)     │    └──────────────────┘    │ - UUID5 Dedupe    │   │ - Grounded Ans. │
└──────────────────┘                            └─────────┬─────────┘   └────────┬────────┘
                                                          │                      │
                                                          ▼                      ▼
                                        ┌─────────────────────────────────────────────────┐
                                        │            Vector Storage & Retrieval           │
                                        │                                                 │
                                        │  • Qdrant (Dense + FastEmbed Sparse RRF)        │
                                        │  • ChromaDB (Dense + Local BM25Okapi RRF)       │
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
│   ├── __init__.py           # Vector client and store exports
│   ├── chain.py              # Reva conversational RAG chain & history contextualization
│   ├── chunking.py           # Language-aware recursive character chunking
│   ├── loader.py             # SmartLoaderManager for multi-format documents, code & git
│   └── vectordb.py           # Unified vector store interface, hybrid search & BM25Okapi
├── routes/
│   ├── __init__.py           # Router exports
│   ├── collections.py        # Collection CRUD and statistics endpoints
│   └── ingestion.py          # File and Git repository ingestion endpoints
├── bm25_store/               # Synchronized BM25Okapi indexes for ChromaDB
├── chroma_vector_db/         # Local Chroma database persistence directory
├── compose.yaml              # Docker Compose service definition (rag-api + qdrant)
├── Dockerfile                # Production multi-stage Docker container (Python 3.14 + uv)
├── main.py                   # FastAPI application entry point & health check endpoints
├── pyproject.toml            # Project dependencies and packaging metadata
├── uv.lock                   # Deterministic dependency lockfile
├── .env.example              # Sample environment configuration template
├── API.md                    # Dedicated complete REST API specification
└── README.md                 # Project documentation
```

---

## ⚙️ Configuration (.env)

Create a `.env` file in the project root. You can copy the template:

```bash
cp .env.example .env
```

### Environment Variables Reference

```dotenv
# ==========================================
# Chat LLM Configuration
# Providers: ollama | private | openai | google
# (Use 'private' for LM Studio or local OpenAI-compatible endpoints)
# ==========================================
chatllm_provider=google
chatllm_base_url=http://localhost:1234/v1/
chatllm_model=gemini-2.5-flash
chatllm_condense_model=gemini-2.5-flash
chatllm_request_timeout=60

# ==========================================
# Embedding Model Configuration
# Providers: ollama | private | openai | google
# ==========================================
embedding_provider=private
embedding_base_url=http://localhost:1234/v1/
embedding_model=text-embedding-embeddinggemma-300m
embedding_size=768
embedding_request_timeout=60

# ==========================================
# Cloud API Keys (Required if using google or openai)
# ==========================================
GOOGLE_API_KEY=your_google_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

# ==========================================
# Vector Database Configuration
# Options: qdrant | chroma
# ==========================================
vector_db_provider=qdrant

# --- Qdrant Settings ---
qdrant_url=http://localhost:6333
qdrant_api_key=None
qdrant_dense_vector_name=dense
qdrant_sparse_vector_name=sparse
sparse_embedding_model=Qdrant/bm25
fastembed_cache_dir=/app/.cache/fastembed
hybrid_prefetch_limit=50
rrf_k=2

# --- ChromaDB & Storage Paths ---
local_path=./
chroma_db_path=./chroma_vector_db/
local_repo_path=./repo/
bm25_store_path=./bm25_store/
matadata_filter_allowed=['file_type', 'file_name']
```

---

## 🚀 Getting Started

### Option 1: Run with Docker Compose (Recommended)

The easiest way to spin up both the **FastAPI RAG API** and **Qdrant Vector Database**:

1. Ensure Docker & Docker Compose are installed.
2. Configure `.env` with your desired model settings and API keys.
3. Start the containers:

```bash
docker compose up --build
```

- API Server: [http://localhost:8000](http://localhost:8000)
- Swagger Docs: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- Qdrant Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

To stop the services:
```bash
docker compose down
```

---

### Option 2: Local Installation

#### Prerequisites

- **Python**: `>= 3.10` (Python 3.14 tested)
- [**uv**](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- A running LLM & embedding provider:
  - [LM Studio](https://lmstudio.ai/) (set provider to `private`)
  - [Ollama](https://ollama.com/) (set provider to `ollama`)
  - Cloud API Key ([Google AI Studio](https://aistudio.google.com/) or [OpenAI](https://platform.openai.com/))
- (Optional) Local Qdrant container if using `vector_db_provider=qdrant`:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
  ```

#### Steps

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
   ```bash
   cp .env.example .env
   # Edit .env with your preferred provider settings
   ```

4. **Start the API Server**:

   Using `uv`:
   ```bash
   uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access Interactive Docs**:
   - Swagger UI: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
   - ReDoc: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)

---

## 📖 API Endpoints Reference

> For full request/response schemas, field types, error status codes, and multi-language client code examples, see the dedicated [**API Documentation (API.md)**](API.md).

### 1. Operational & System Health

#### Root Info
```http
GET /api/
```
**Response**:
```json
{
  "message": "LangChain RAG API",
  "version": "1.0.0"
}
```

#### Health Status
Checks operational connectivity across API, Chat LLM, Embedding Model, and Vector DB concurrently:
```http
GET /api/health
```
**Response**:
```json
{
  "api_services": true,
  "chat_LLM_services": true,
  "embedding_Model_services": true,
  "vectordb_services": true
}
```

---

### 2. Collection Management

#### List Collections
```http
GET /api/collections/
```
**Response**:
```json
{
  "collections": ["research_papers", "codebase_docs"]
}
```

#### Get Collection Details & Stats
```http
GET /api/collections/{collection_name}
```
**Response**:
```json
{
  "name": "research_papers",
  "status": "active",
  "total_chunks": 142,
  "files_count": 3
}
```

#### Create Collection
```http
POST /api/collections/
Content-Type: application/json

{
  "name": "research_papers"
}
```
**Response**:
```json
{
  "message": "Created Successfully"
}
```

#### Delete Collection
```http
DELETE /api/collections/
Content-Type: application/json

{
  "name": "research_papers"
}
```
**Response**:
```json
{
  "message": "Collection research_papers deleted successfully."
}
```

---

### 3. Document Ingestion

#### Inspect Collection Files & Chunk Distribution
```http
GET /api/upload/{collection_name}
```
**Response**:
```json
{
  "total_chunks": 42,
  "filenames": ["attention_is_all_you_need.pdf"],
  "files": [
    {
      "file_name": "attention_is_all_you_need.pdf",
      "chunks": 42
    }
  ]
}
```

#### Upload Files
Upload one or multiple local documents or code files:
```bash
curl -X POST "http://localhost:8000/api/upload/files/research_papers" \
     -F "files=@/path/to/paper.pdf" \
     -F "files=@/path/to/pipeline.py"
```
**Response**:
```json
{
  "collection_name": "research_papers",
  "files_received": 2,
  "chunks_created": 68,
  "ingest_status": "Documents added successfully"
}
```

#### Ingest Public Git Repository
Clone and ingest any public Git repository directly by URL:
```bash
curl -X POST "http://localhost:8000/api/upload/git" \
     -H "Content-Type: application/json" \
     -d '{
       "collection_name": "codebase_docs",
       "git_url": "https://github.com/tiangolo/fastapi.git"
     }'
```
**Response**:
```json
{
  "collection_name": "codebase_docs",
  "files_received": 350,
  "chunks_created": 1820,
  "ingest_status": "Documents added successfully"
}
```

#### Delete Chunks by File Name
Remove all vector chunks corresponding to a specific source document:
```bash
curl -X DELETE "http://localhost:8000/api/upload/delete" \
     -H "Content-Type: application/json" \
     -d '{
       "collection_name": "research_papers",
       "file_name": "paper.pdf"
     }'
```
**Response**:
```json
{
  "message": "total 42 chunks deleted "
}
```

---

### 4. Conversational RAG Chat (`Reva`)

Ask questions against any collection with optional conversation history. Queries with history are automatically reformulated into standalone queries before retrieval:

```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "collection_name": "research_papers",
       "query": "What architecture is proposed in this paper?",
       "history": [
         ["user", "Hi!"],
         ["ai", "Hello! I am Reva, how can I assist you with your data or codebase today?"]
       ]
     }'
```

**Response**:
```json
{
  "airesponse": "The paper proposes the Transformer architecture, which relies entirely on self-attention mechanisms without recurrent or convolutional neural networks.\n\n(Filename: attention_is_all_you_need.pdf, Page: 2)",
  "history": [
    ["user", "Hi!"],
    ["ai", "Hello! I am Reva, how can I assist you with your data or codebase today?"],
    ["user", "What architecture is proposed in this paper?"],
    ["ai", "The paper proposes the Transformer architecture, which relies entirely on self-attention mechanisms without recurrent or convolutional neural networks.\n\n(Filename: attention_is_all_you_need.pdf, Page: 2)"]
  ]
}
```

---

## 🛠️ Development & Testing

Run unit tests with `pytest`:

```bash
uv run pytest
```

Run test suite with coverage report:
```bash
uv run pytest --cov=core --cov=config --cov=routes
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
