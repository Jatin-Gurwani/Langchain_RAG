# LangChain RAG System — API Reference Documentation 📡

This document provides a comprehensive technical specification of all REST endpoints exposed by the **LangChain RAG API**.

- **Base URL**: `http://localhost:8000`
- **Interactive Swagger UI**: [`http://localhost:8000/api/docs`](http://localhost:8000/api/docs)
- **Interactive ReDoc**: [`http://localhost:8000/api/redoc`](http://localhost:8000/api/redoc)
- **Standard Payload Format**: `application/json` (except multi-file uploads which use `multipart/form-data`)

---

## 📑 Table of Contents

1. [API Overview & Conventions](#-api-overview--conventions)
2. [Endpoints Summary](#-endpoints-summary)
3. [Operational Endpoints](#1-operational-endpoints)
   - [GET /api/](#get-api)
   - [GET /api/health](#get-apihealth)
4. [Collection Management Endpoints](#2-collection-management-endpoints)
   - [GET /api/collections/](#get-apicollections)
   - [GET /api/collections/{collection_name}](#get-apicollectionscollection_name)
   - [POST /api/collections/](#post-apicollections)
   - [DELETE /api/collections/](#delete-apicollections)
5. [Document Ingestion Endpoints](#3-document-ingestion-endpoints)
   - [GET /api/upload/{collection_name}](#get-apiuploadcollection_name)
   - [POST /api/upload/files/{collection_name}](#post-apiuploadfilescollection_name)
   - [POST /api/upload/git](#post-apiuploadgit)
   - [DELETE /api/upload/delete](#delete-apiuploaddelete)
6. [Conversational RAG Chat Endpoints](#4-conversational-rag-chat-endpoints)
   - [POST /api/chat](#post-apichat)
7. [Common HTTP Status Codes](#-common-http-status-codes)
8. [Code Examples](#-code-examples)

---

## 🌐 API Overview & Conventions

### Base URL & Versioning
All endpoints are namespaced under the `/api` prefix. The root endpoint provides service status and version information.

### Error Handling
All errors return a JSON response containing a `detail` message explaining the cause:

```json
{
  "detail": "Collection name is incorrect or does not exist"
}
```

---

## 📊 Endpoints Summary

| Method | Endpoint | Tag | Summary |
|---|---|---|---|
| `GET` | `/api/` | `operational` | Service information and root check |
| `GET` | `/api/health` | `operational` | Concurrent health status of API, LLM, Embeddings, & Vector DB |
| `GET` | `/api/collections/` | `collections` | List all existing collections |
| `GET` | `/api/collections/{collection_name}` | `collections` | Get collection status, chunk count, and file count |
| `POST` | `/api/collections/` | `collections` | Create a new vector collection |
| `DELETE` | `/api/collections/` | `collections` | Delete an existing vector collection |
| `GET` | `/api/upload/{collection_name}` | `documents` | Inspect ingested files and chunk distribution in a collection |
| `POST` | `/api/upload/files/{collection_name}` | `documents` | Upload and chunk local documents and code files |
| `POST` | `/api/upload/git` | `documents` | Clone and ingest a public Git repository |
| `DELETE` | `/api/upload/delete` | `documents` | Delete all chunks originating from a specific file |
| `POST` | `/api/chat` | `Chat` | Ask questions with conversation history and contextual retrieval |

---

## 1. Operational Endpoints

### `GET /api/`
Returns basic service information and API version.

#### Request
- **Headers**: None required

#### Responses
- **200 OK**:
  ```json
  {
    "message": "LangChain RAG API",
    "version": "1.0.0"
  }
  ```

#### Example
```bash
curl -X GET "http://localhost:8000/api/"
```

---

### `GET /api/health`
Concurrently verifies operational connectivity for:
1. **Chat LLM**: Sends an asynchronous single-token ping probe to the configured chat model (10s timeout).
2. **Embedding Model**: Asynchronously embeds a ping string (5s timeout).
3. **Vector Database**: Probes Qdrant or ChromaDB connection (5s timeout).

#### Request
- **Headers**: None required

#### Responses
- **200 OK**:
  ```json
  {
    "api_services": true,
    "chat_LLM_services": true,
    "embedding_Model_services": true,
    "vectordb_services": true
  }
  ```

#### Example
```bash
curl -X GET "http://localhost:8000/api/health"
```

---

## 2. Collection Management Endpoints

### `GET /api/collections/`
Retrieves a list of all existing collections available in the active vector database (Qdrant or ChromaDB).

#### Request
- **Headers**: None required

#### Responses
- **200 OK**:
  ```json
  {
    "collections": [
      "research_papers",
      "codebase_docs",
      "legal_briefs"
    ]
  }
  ```
- **503 Service Unavailable**: Failed to fetch collections from backend.

#### Example
```bash
curl -X GET "http://localhost:8000/api/collections/"
```

---

### `GET /api/collections/{collection_name}`
Provides statistics and active status for a specific collection.

#### Request
- **Path Parameters**:
  - `collection_name` (`string`, required): Case-insensitive name of the collection.

#### Responses
- **200 OK**:
  ```json
  {
    "name": "research_papers",
    "status": "active",
    "total_chunks": 142,
    "files_count": 3
  }
  ```
- **404 Not Found**: Collection is incorrect or does not exist.
- **503 Service Unavailable**: Exception interacting with the vector backend.

#### Example
```bash
curl -X GET "http://localhost:8000/api/collections/research_papers"
```

---

### `POST /api/collections/`
Creates a new collection in the active vector database. For Qdrant, configures both dense cosine vectors and sparse vectors for hybrid search.

#### Request
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "name": "string"
  }
  ```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | Yes | Unique name for the new collection |

#### Responses
- **200 OK**:
  ```json
  {
    "message": "Created Successfully"
  }
  ```
- **403 Forbidden**: Collection `{name}` already exists.
- **503 Service Unavailable**: Unable to create new collection in the backend.

#### Example
```bash
curl -X POST "http://localhost:8000/api/collections/" \
     -H "Content-Type: application/json" \
     -d '{"name": "research_papers"}'
```

---

### `DELETE /api/collections/`
Deletes an entire collection and all associated vectors and BM25 index data.

#### Request
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "name": "string"
  }
  ```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | Yes | Name of the collection to delete |

#### Responses
- **200 OK**:
  ```json
  {
    "message": "Collection research_papers deleted successfully."
  }
  ```
- **403 Forbidden**: Collection `{name}` does not exist.
- **503 Service Unavailable**: Unable to delete collection at this time.

#### Example
```bash
curl -X DELETE "http://localhost:8000/api/collections/" \
     -H "Content-Type: application/json" \
     -d '{"name": "research_papers"}'
```

---

## 3. Document Ingestion Endpoints

### `GET /api/upload/{collection_name}`
Inspects chunk distribution and lists all individual files stored within a collection.

#### Request
- **Path Parameters**:
  - `collection_name` (`string`, required): Name of the collection.

#### Responses
- **200 OK**:
  ```json
  {
    "total_chunks": 42,
    "filenames": [
      "transformer_paper.pdf",
      "model_utils.py"
    ],
    "files": [
      {
        "file_name": "transformer_paper.pdf",
        "chunks": 32
      },
      {
        "file_name": "model_utils.py",
        "chunks": 10
      }
    ]
  }
  ```
- **404 Not Found**: Collection name is incorrect or does not exist.

#### Example
```bash
curl -X GET "http://localhost:8000/api/upload/research_papers"
```

---

### `POST /api/upload/files/{collection_name}`
Uploads one or more local files, parses them with format-specific loaders, generates syntax-aware chunks, creates deterministic UUID5 IDs, and stores them in the vector database.

#### Supported File Types
- **Documents**: `.pdf`, `.docx`, `.pptx`, `.ppt`, `.md`, `.mdx`, `.csv`, `.txt`, `.rst`, `.log`
- **Notebooks**: `.ipynb`
- **Code & Configs**: `.py`, `.js`, `.jsx`, `.ts`, `.tsx`, `.java`, `.c`, `.cc`, `.cpp`, `.cs`, `.html`, `.css`, `.json`, `.yaml`, `.yml`, `.sql`, `.sh`, `.bash`, `.zsh`, `.toml`, `.ini`, `.cfg`, `.dockerfile`

#### Request
- **Path Parameters**:
  - `collection_name` (`string`, required): Target collection.
- **Headers**: `Content-Type: multipart/form-data`
- **Form Data**:
  - `files` (`binary[]`, required): One or more files to upload.

#### Responses
- **200 OK**:
  ```json
  {
    "collection_name": "research_papers",
    "files_received": 2,
    "chunks_created": 68,
    "ingest_status": "Documents added successfully"
  }
  ```
- **400 Bad Request**: No files provided.
- **404 Not Found**: Collection name is incorrect or does not exist.
- **500 Internal Server Error**: Internal error while adding chunks to vector DB.

#### Example
```bash
curl -X POST "http://localhost:8000/api/upload/files/research_papers" \
     -F "files=@paper.pdf" \
     -F "files=@helpers.py"
```

---

### `POST /api/upload/git`
Clones a public Git repository into a temporary directory, automatically filters out binary/unsupported extensions, chunks supported code/doc files, and indexes them into the collection.

#### Request
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "collection_name": "string",
    "git_url": "string"
  }
  ```

| Field | Type | Required | Description |
|---|---|---|---|
| `collection_name` | `string` | Yes | Target collection name |
| `git_url` | `string` | Yes | HTTP URL ending in `.git` (e.g. `https://github.com/fastapi/fastapi.git`) |

#### Responses
- **200 OK**:
  ```json
  {
    "collection_name": "codebase_docs",
    "files_received": 142,
    "chunks_created": 845,
    "ingest_status": "Documents added successfully"
  }
  ```
- **404 Not Found**: Collection name is incorrect or does not exist.
- **504 Gateway Timeout**: Git URL is not valid (must start with `http` and end with `.git`).
- **500 Internal Server Error**: Internal error while adding chunks to vector DB.

#### Example
```bash
curl -X POST "http://localhost:8000/api/upload/git" \
     -H "Content-Type: application/json" \
     -d '{
       "collection_name": "codebase_docs",
       "git_url": "https://github.com/tiangolo/fastapi.git"
     }'
```

---

### `DELETE /api/upload/delete`
Deletes all chunks associated with a specific file name from a collection. Also synchronizes local BM25 indices if ChromaDB is active.

#### Request
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "collection_name": "string",
    "file_name": "string",
    "file_type": "string"
  }
  ```

| Field | Type | Required | Description |
|---|---|---|---|
| `collection_name` | `string` | Yes | Target collection name |
| `file_name` | `string` | Yes | Exact file name to remove (e.g., `paper.pdf`) |
| `file_type` | `string` | No | Optional extension filter (e.g., `.pdf`) |

#### Responses
- **200 OK**:
  ```json
  {
    "message": "total 42 chunks deleted "
  }
  ```
- **404 Not Found**: Collection name is incorrect or chunks related to file do not exist.
- **503 Service Unavailable**: File name is not found in the vector database.

#### Example
```bash
curl -X DELETE "http://localhost:8000/api/upload/delete" \
     -H "Content-Type: application/json" \
     -d '{
       "collection_name": "research_papers",
       "file_name": "paper.pdf"
     }'
```

---

## 4. Conversational RAG Chat Endpoints

### `POST /api/chat`
Answers questions grounded strictly in the context stored in the target collection.

#### Behavior & Persona (`Reva`)
1. **History Rewriting**: If prior chat messages are provided in `history`, a standalone search query is reformulated via `RunnableBranch` before vector retrieval.
2. **Hybrid RRF Retrieval**: Performs semantic dense search and BM25 lexical search, combining ranks using Reciprocal Rank Fusion (`rrf_k`).
3. **Citations**: Citations are formatted as `(Filename: <name>, Page: <num>)` at the end of the response when available. Internal document IDs and raw scores are never exposed.
4. **Code Queries**: When requested for code snippets, responds with clean code blocks only (no conversational intro/outro or explanatory comments).
5. **Context Grounding**: If the answer cannot be found in the retrieved context, responds strictly: `"We don't have any knowledge related to [topic]."`.

#### Request
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "collection_name": "string",
    "query": "string",
    "history": [
      ["user", "string"],
      ["ai", "string"]
    ]
  }
  ```

| Field | Type | Required | Description |
|---|---|---|---|
| `collection_name` | `string` | Yes | Target collection to query against |
| `query` | `string` | Yes | User's question or prompt |
| `history` | `list[tuple[str, str]]` | No | Sequential chat history pairs `[role, content]` (`"user"` or `"ai"`) |

#### Responses
- **200 OK**:
  ```json
  {
    "airesponse": "The Transformer uses multi-head self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.\n\n(Filename: attention_is_all_you_need.pdf, Page: 3)",
    "history": [
      ["user", "What mechanism does the model use?"],
      ["ai", "The Transformer uses multi-head self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.\n\n(Filename: attention_is_all_you_need.pdf, Page: 3)"]
    ]
  }
  ```
- **404 Not Found**: Collection name is incorrect or does not exist.
- **503 Service Unavailable**: Unable to process query at this time.

#### Example
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "collection_name": "research_papers",
       "query": "What mechanism does the model use?",
       "history": []
     }'
```

---

## 🚦 Common HTTP Status Codes

| Status Code | Meaning | Common Scenario in LangChain RAG |
|---|---|---|
| **200 OK** | Success | Operation completed successfully (data fetched, collection created/deleted, query answered). |
| **400 Bad Request** | Invalid Input | Uploading empty file batch or missing files payload. |
| **403 Forbidden** | Conflict / Duplicate | Attempting to create a collection that already exists, or deleting a non-existent one. |
| **404 Not Found** | Resource Missing | Collection name does not exist, or targeted file chunks not found. |
| **500 Internal Error** | Processing Failure | Chunking failure or error during vector insertion. |
| **503 Unavailable** | Backend Failure | LLM timeout, embedding service offline, or vector database unreachable. |
| **504 Gateway Timeout** | Format / Network Failure | Git URL is invalid (missing `http` prefix or `.git` suffix) or remote clone timed out. |

---

## 💡 Code Examples

### Python (`requests`)
```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Health Check
health = requests.get(f"{BASE_URL}/api/health").json()
print("Health:", health)

# 2. Create Collection
create_res = requests.post(f"{BASE_URL}/api/collections/", json={"name": "ai_docs"}).json()
print("Create Collection:", create_res)

# 3. Chat with History
chat_payload = {
    "collection_name": "ai_docs",
    "query": "Summarize the key contribution.",
    "history": []
}
chat_res = requests.post(f"{BASE_URL}/api/chat", json=chat_payload).json()
print("Reva Response:", chat_res["airesponse"])
```

### JavaScript (`fetch`)
```javascript
const BASE_URL = "http://localhost:8000";

async function queryRAG() {
  const response = await fetch(`${BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      collection_name: "ai_docs",
      query: "How does the attention mechanism work?",
      history: []
    })
  });
  const data = await response.json();
  console.log("Reva:", data.airesponse);
}

queryRAG();
```
