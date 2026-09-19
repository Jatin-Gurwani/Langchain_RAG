from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_chroma import Chroma
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse, ApiException
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    FieldCondition, MatchValue, Filter, FilterSelector,
    Prefetch, FusionQuery, Fusion, SparseVector,
)
import chromadb
from functools import lru_cache
from config import get_settings, get_embeddingmodel
from typing import List, Literal
from langchain_core.documents import Document
from uuid import uuid5, NAMESPACE_URL
from tqdm import tqdm
from os import makedirs
from pathlib import Path
import pickle
import re
from rank_bm25 import BM25Okapi

settings = get_settings()

DBType = Literal["qdrant", "chroma"]

_WORD_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, check_compatibility=False)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    cpath = settings.chroma_db_path
    makedirs(cpath, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_db_path)


def _active_db_type() -> DBType:
    db_type = settings.vector_db_provider
    if db_type not in ("qdrant", "chroma"):
        raise ValueError(
            f"Invalid vector_db_provider '{db_type}' in settings. Valid options are: qdrant, chroma"
        )
    return db_type


def _get_client():
    db_type = _active_db_type()
    if db_type == "qdrant":
        return get_qdrant_client()
    elif db_type == "chroma":
        return get_chroma_client()


# Backward-compat alias: existing callers using get_client() keep working (Qdrant only).
def get_client() -> QdrantClient:
    return get_qdrant_client()


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_collection_list() -> List[str]:
    db_type = _active_db_type()
    if db_type == "qdrant":
        client = get_qdrant_client()
        return [c.name for c in client.get_collections().collections]
    elif db_type == "chroma":
        client = get_chroma_client()
        # chromadb >= 0.4 returns Collection objects; older versions return plain names.
        collections = client.list_collections()
        return [c.name if hasattr(c, "name") else c for c in collections]


def get_collection_data_by_name(name: str):
    name = name.lower()
    try:
        client = _get_client()
        data = client.get_collection(name)
        return [200, data]
    except UnexpectedResponse as e:
        return [e.status_code, f"collection {name} is incorrect or does not exist"]
    except ApiException as e:
        print(f"Error in get_collection_data_by_name: {e}")
        return [503, "Exception while interacting with backend"]
    except Exception as e:
        return [404, f"collection {name} is incorrect or does not exist ({e})"]


def create_collection(name: str):
    name = name.lower()
    db_type = _active_db_type()
    if name in get_collection_list():
        return [403, f"Collection {name} already exists."]
    try:
        if db_type == "qdrant":
            client = get_qdrant_client()
            result = client.create_collection(
                collection_name=name,
                vectors_config={
                    settings.qdrant_dense_vector_name: VectorParams(
                        size=settings.embedding_size, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    settings.qdrant_sparse_vector_name: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            if not result:
                return [503, "unable to create new collection"]
            return [200, "Created Successfully"]

        elif db_type == "chroma":
            client = get_chroma_client()
            client.create_collection(name=name)
            return [200, "Created Successfully"]  # normalized "success" status, mirroring Qdrant's CollectionStatus.GREEN
    except Exception:
        return [503, "unable to create new collection"]


def delete_collection(name: str):
    name = name.lower()
    db_type = _active_db_type()
    if name not in get_collection_list():
        return [403, f"Collection {name} does not exist."]
    try:
        if db_type == "qdrant":
            client = get_qdrant_client()
            result = client.delete_collection(name)
            if not result:
                return [503, "Unable to delete collection at this movement"]
        elif db_type == "chroma":
            client = get_chroma_client()
            client.delete_collection(name)
        return [200, f"Collection {name} deleted successfully."]
    except Exception:
        return [503, "Unable to delete collection at this movement"]


def get_vector_db(db_name: str, embedding: object | None = None) -> QdrantVectorStore | Chroma:
    db_name = db_name.lower()
    db_type = _active_db_type()
    if db_name not in get_collection_list():
        return f"Collection {db_name} does not exist."

    if not embedding:
        embedding = get_embeddingmodel()

    if db_type == "qdrant":
        client = get_qdrant_client()
        sparse_embedding = FastEmbedSparse(model_name=settings.sparse_embedding_model)
        return QdrantVectorStore(
            client=client,
            collection_name=db_name,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=settings.qdrant_dense_vector_name,
            sparse_vector_name=settings.qdrant_sparse_vector_name,
        )

    elif db_type == "chroma":
        client = get_chroma_client()
        # Dense-only in Chroma itself: hybrid (BM25) is layered on top at
        # retrieval time — see _retrieve_chroma_hybrid.
        return Chroma(
            client=client,
            collection_name=db_name,
            embedding_function=embedding,
        )


# ---------------------------------------------------------------------------
# Ingestion (already backend-agnostic via LangChain's VectorStore interface)
# ---------------------------------------------------------------------------

def _deterministic_chunk_id(chunk: Document) -> str:
    file_name = chunk.metadata.get("file_name", "")
    key = f"{file_name}|{chunk.page_content}"
    return str(uuid5(NAMESPACE_URL, key))


def _bm25_store_path(collection_name: str) -> Path:
    makedirs(settings.bm25_store_path, exist_ok=True)
    return Path(settings.bm25_store_path) / f"{collection_name}.pkl"


def _load_bm25_store(collection_name: str) -> dict:
    path = _bm25_store_path(collection_name)
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def _save_bm25_store(collection_name: str, store: dict) -> None:
    path = _bm25_store_path(collection_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(store, f)
    tmp.replace(path)  # atomic swap, avoids a half-written file on crash
    # NOTE: read-modify-write with no locking — fine for sequential/single-
    # worker ingestion, but concurrent writers to the same collection can
    # clobber each other. Add a file lock if that becomes a real scenario.


def add_chunks(db: QdrantVectorStore | Chroma, chunks: List[Document], batch_size: int = 50):
    try:
        chunks_id = [_deterministic_chunk_id(c) for c in chunks]
        total_itr = 0
        for batch_itr in tqdm(range(0, len(chunks), batch_size), desc='adding chunks in vector db'):
            batch_chunks = chunks[batch_itr:batch_itr + batch_size]
            batch_chunks_id = chunks_id[batch_itr:batch_itr + batch_size]
            _ = db.add_documents(batch_chunks, ids=batch_chunks_id)
            total_itr += 1

        if isinstance(db, Chroma):
            # Keep the BM25 pickle in sync — it's what _retrieve_chroma_hybrid
            # reads from instead of the live collection.
            collection_name = db._collection.name
            store = _load_bm25_store(collection_name)
            store.update({cid: (c.page_content, c.metadata) for cid, c in zip(chunks_id, chunks)})
            _save_bm25_store(collection_name, store)

        return [200, "Documents added successfully"]
    except Exception as e:
        print("Exception at adding chunks in vector db : ", e)
        return [500, "Internal error while adding chunks in vector db"]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _retrieve_hybrid_rrf(db: QdrantVectorStore, query: str, k: int, qdrant_filter: Filter | None):
    dense_query = db.embeddings.embed_query(query)
    sparse_query = db.sparse_embeddings.embed_query(query)
    prefetch_limit = max(k * 4, settings.hybrid_prefetch_limit)

    points = db.client.query_points(
        collection_name=db.collection_name,
        prefetch=[
            Prefetch(using=db.vector_name, query=dense_query, filter=qdrant_filter, limit=prefetch_limit),
            Prefetch(
                using=db.sparse_vector_name,
                query=SparseVector(indices=sparse_query.indices, values=sparse_query.values),
                filter=qdrant_filter,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
        with_vectors=False,
    ).points

    return [
        (
            db._document_from_point(point, db.collection_name, db.content_payload_key, db.metadata_payload_key),
            point.score,
        )
        for point in points
    ]


def _matches_where(metadata: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(metadata.get(k) == v for cond in where["$and"] for k, v in cond.items())
    return all(metadata.get(k) == v for k, v in where.items())


def _retrieve_chroma_hybrid(db: Chroma, query: str, k: int, where: dict | None):
    """Dense (Chroma) + sparse (BM25) retrieval, fused with Reciprocal Rank
    Fusion. Chroma has no native hybrid mode, so BM25 runs client-side,
    reading chunks from the pickle add_chunks/delete_chunks keep in sync
    rather than paging through the live collection on every query.
    """
    collection_name = db._collection.name
    store = _load_bm25_store(collection_name)
    entries = [(doc_id, doc, meta) for doc_id, (doc, meta) in store.items() if _matches_where(meta, where)]
    if not entries:
        return []
    ids, docs, metas = zip(*entries)

    prefetch_limit = max(k * 4, settings.hybrid_prefetch_limit)

    dense_ranking = db._collection.query(
        query_embeddings=[db.embeddings.embed_query(query)],
        n_results=min(prefetch_limit, len(ids)),
        where=where,
    )["ids"][0]

    bm25 = BM25Okapi([_WORD_RE.findall(doc.lower()) for doc in docs])
    bm25_scores = bm25.get_scores(_WORD_RE.findall(query.lower()))
    sparse_ranking = [
        ids[i] for i in sorted(range(len(ids)), key=lambda i: bm25_scores[i], reverse=True)[:prefetch_limit]
    ]

    fused: dict = {}
    for ranking in (dense_ranking, sparse_ranking):
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)

    lookup = dict(zip(ids, zip(docs, metas)))
    top_ids = [doc_id for doc_id in sorted(fused, key=fused.get, reverse=True) if doc_id in lookup][:k]
    return [(Document(page_content=lookup[i][0], metadata=lookup[i][1] or {}), fused[i]) for i in top_ids]


def retrieve_chunks(db: QdrantVectorStore | Chroma, query: str, k: int = 5, filter: dict = False):
    qdrant_filter = None
    chroma_filter = None

    if filter:
        invalid_keys = [key for key in filter.keys() if key not in settings.matadata_filter_allowed]
        if invalid_keys:
            return f"Filter keys should be in {settings.matadata_filter_allowed}"

        if isinstance(db, QdrantVectorStore):
            # NOTE: langchain-qdrant nests metadata under "metadata.<key>" in the point payload.
            qdrant_filter = Filter(must=[
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in filter.items()
            ])
        elif isinstance(db, Chroma):
            # NOTE: langchain-chroma stores metadata flat (no "metadata." prefix).
            chroma_filter = (
                dict(filter) if len(filter) == 1
                else {"$and": [{key: value} for key, value in filter.items()]}
            )
        else:
            raise ValueError(f"Unsupported vector store type: {type(db)}")

    if isinstance(db, QdrantVectorStore) and db.retrieval_mode == RetrievalMode.HYBRID:
        return _retrieve_hybrid_rrf(db, query, k, qdrant_filter)

    if isinstance(db, QdrantVectorStore):
        return db.similarity_search_with_score(query, k, filter=qdrant_filter)

    return _retrieve_chroma_hybrid(db, query, k, chroma_filter)


def get_collection_stats(collection_name: str, file_name: str | None = None) -> dict:
    """Return total stored chunks and unique filenames for a collection.

    When file_name is provided, the response is limited to that file only.
    The payload shape is:
    {
        "total_chunks": int,
        "filenames": list[str],
        "files": [{"file_name": str, "chunks": int}]
    }
    """
    collection_name = collection_name.lower()
    db_type = _active_db_type()

    if db_type == "qdrant":
        client = get_qdrant_client()
        try:
            client.get_collection(collection_name)
        except Exception:
            return {"total_chunks": 0, "filenames": [], "files": []}

        file_filter = None
        if file_name:
            file_filter = Filter(
                must=[FieldCondition(key="metadata.file_name", match=MatchValue(value=file_name))]
            )

        total_chunks = client.count(
            collection_name=collection_name,
            count_filter=file_filter,
            exact=True,
        ).count

        if total_chunks == 0:
            if file_name:
                return {"total_chunks": 0, "filenames": [file_name], "files": [{"file_name": file_name, "chunks": 0}]}
            return {"total_chunks": 0, "filenames": [], "files": []}

        file_map = {}
        unique_files = set()
        offset = None

        while True:
            result = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=100,
                with_payload=["metadata"],
                with_vectors=False,
                scroll_filter=file_filter,
            )
            points, next_offset = result
            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                current_file = payload.get("file_name")
                if not current_file and isinstance(metadata, dict):
                    current_file = metadata.get("file_name")
                if not current_file:
                    continue
                unique_files.add(current_file)
                file_map[current_file] = file_map.get(current_file, 0) + 1

            if not next_offset:
                break
            offset = next_offset

        filenames = sorted(unique_files)
        files_payload = [{"file_name": name, "chunks": file_map.get(name, 0)} for name in filenames]
        return {"total_chunks": total_chunks, "filenames": filenames, "files": files_payload}

    elif db_type == "chroma":
        client = get_chroma_client()
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            return {"total_chunks": 0, "filenames": [], "files": []}

        where = {"file_name": file_name} if file_name else None
        if where is None:
            total_chunks = collection.count()
        else:
            total_chunks = 0
            offset = 0
            while True:
                kwargs = {"include": ["metadatas"], "limit": 100, "offset": offset, "where": where}
                result = collection.get(**kwargs)
                ids = result.get("ids", [])
                total_chunks += len(ids)
                if len(ids) < 100:
                    break
                offset += len(ids)

        if total_chunks == 0:
            if file_name:
                return {"total_chunks": 0, "filenames": [file_name], "files": [{"file_name": file_name, "chunks": 0}]}
            return {"total_chunks": 0, "filenames": [], "files": []}

        file_map = {}
        unique_files = set()
        offset = 0

        while True:
            kwargs = {"include": ["metadatas"], "limit": 100, "offset": offset}
            if where is not None:
                kwargs["where"] = where
            result = collection.get(**kwargs)
            ids = result.get("ids", [])
            if not ids:
                break

            for metadata in result.get("metadatas", []):
                if metadata and "file_name" in metadata and metadata["file_name"]:
                    current_file = metadata["file_name"]
                    unique_files.add(current_file)
                    file_map[current_file] = file_map.get(current_file, 0) + 1

            if len(ids) < 100:
                break
            offset += len(ids)

        filenames = sorted(unique_files)
        files_payload = [{"file_name": name, "chunks": file_map.get(name, 0)} for name in filenames]
        return {"total_chunks": total_chunks, "filenames": filenames, "files": files_payload}

    return {"total_chunks": 0, "filenames": [], "files": []}


# ---------------------------------------------------------------------------
# Deletion by source file
# ---------------------------------------------------------------------------

def delete_chunks(collection_name: str, file_name: str, file_type: str | bool = False) -> int:

    collection_name = collection_name.lower()
    db_type = _active_db_type()

    if db_type == "qdrant":
        client = get_qdrant_client()
        conditions = [FieldCondition(key="metadata.file_name", match=MatchValue(value=file_name))]
        if file_type:
            conditions.append(FieldCondition(key="metadata.file_type", match=MatchValue(value=file_type)))
        delete_filter = Filter(must=conditions)

        matched = client.count(
            collection_name=collection_name, count_filter=delete_filter, exact=True
        ).count
        if matched == 0:
            return 0

        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=delete_filter),
        )
        return matched

    elif db_type == "chroma":
        client = get_chroma_client()
        collection = client.get_collection(collection_name)

        if file_type:
            where = {"$and": [{"file_name": file_name}, {"file_type": file_type}]}
        else:
            where = {"file_name": file_name}

        matched_ids = collection.get(where=where).get("ids", [])
        if not matched_ids:
            return 0

        collection.delete(where=where)

        # Keep the BM25 pickle in sync with what's actually still in Chroma.
        store = _load_bm25_store(collection_name)
        for doc_id in matched_ids:
            store.pop(doc_id, None)
        _save_bm25_store(collection_name, store)

        return len(matched_ids)
