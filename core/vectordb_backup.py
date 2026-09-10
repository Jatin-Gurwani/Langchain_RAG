from langchain_qdrant import QdrantVectorStore, RetrievalMode,FastEmbedSparse
from langchain_chroma import Chroma
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse,ApiException
from qdrant_client.http.models import Distance, VectorParams,SparseVectorParams,SparseIndexParams,FieldCondition,MatchValue,Filter,FilterSelector
from functools import lru_cache
from config import get_settings,get_embeddingmodel
from typing import List
from langchain_core.documents import Document
from uuid import uuid4
from tqdm import tqdm

settings = get_settings()

@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, check_compatibility=False)


def get_collection_list()->  List[str]:
    client= get_client()
    return [c.name for c in client.get_collections().collections]

def get_collection_data_by_name(name:str):
    try:
        client= get_client()
        data = client.get_collection(name)
        return [200,data]
    except UnexpectedResponse as e:
        return [e.status_code,f"collection {name} is incorrect or does not exist"]
    except ApiException as e:
        print(f"Error at {__annotations__} : {e}")
        return [503,f"Exception while interacting with backend"]

def create_collection(name:str)-> str:
    client=get_client()
    if name.lower() in get_collection_list():
        return f"Collection {name.lower()} already exists."
    try:
        result = client.create_collection(
            collection_name=name.lower(),
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
        if not result : 
            raise Exception
        else :
            return client.get_collection(name.lower()).status
    except Exception as e:
        return False
    
def delete_collection(name:str)-> str:
    client=get_client()
    if not name.lower() in get_collection_list():
        return f"Collection {name.lower()} does not exists."
    try:
        result = client.delete_collection(name.lower())
        if not result : 
            raise Exception
        else :
            return f"Collection {name.lower()} deleted successfully."
    except Exception as e:
        return False
def get_vector_db(db_name:str,embedding:object) -> QdrantVectorStore|Chroma:
    client = get_client()
    if not  db_name.lower() in get_collection_list():
        return f"Collection {db_name.lower()} does not  exists."
    sparse_embedding = FastEmbedSparse(model_name=settings.sparse_embedding_model)
    return QdrantVectorStore(client= client,
                             collection_name=db_name.lower(),
                             embedding=embedding,
                             sparse_embedding=sparse_embedding,
                             retrieval_mode=RetrievalMode.HYBRID,
                             vector_name=settings.qdrant_dense_vector_name,
                             sparse_vector_name=settings.qdrant_sparse_vector_name
                             )
def add_chunks(db: QdrantVectorStore|Chroma,chunks:List[Document], batch_size:int=50):
    chunks_id = [str(uuid4()) for _ in range( len(chunks))]
    total_itr = 0
    for batch_itr in tqdm(range(0,len(chunks),batch_size),desc='adding chunks in vector db'):
        batch_chunks = chunks[batch_itr:batch_itr+batch_size]
        batch_chunks_id = chunks_id[batch_itr:batch_itr+batch_size]
        _=db.add_documents(batch_chunks,ids=batch_chunks_id,)
        total_itr += 1
    print(total_itr)
def retrieve_chunks(db: QdrantVectorStore|Chroma , query:str,k:int=5,filter:dict=False):
    if filter:
        if False in [f.keys in settings.matadata_filter_allowed for f in filter]:
            return f"Filter keys should be in {settings.matadata_filter_allowed}"
        else :
            f_conditions = [FieldCondition(key=f"metadata.{f[0]}", match=MatchValue(value=f[1])) for f in filter.items()]
            results = db.similarity_search_with_score(query,k,filter=Filter(must=f_conditions))
            return results
    results =db.similarity_search_with_score(query,k)
    return results

def delete_chunks(collection_name:str,file_name: str, file_type: str |bool=False) -> int:
    """Delete every chunk belonging to one ingested source (a URL, a repo
    file, an uploaded document/image). Returns how many chunks were
    removed, 0 if nothing matched."""

    client = get_client()
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

