import asyncio
from fastapi.routing import APIRouter
from fastapi import UploadFile,File,HTTPException,Form
import tempfile
import shutil
from core.vectordb import add_chunks,delete_chunks,get_collection_list,get_vector_db,get_collection_stats
from core.loader import SmartLoaderManager
from core.chunking import recursive_chunking
from config.api_schema import default_response,ingest_created_response,FileUpload,ingest_request_model,delete_chunks_request_model,ingest_data_response
from typing import List,Optional,Annotated
from pathlib import Path

router = APIRouter(prefix="/api/upload",tags=["documents"])

@router.get("/{collection_name}",response_model=ingest_data_response)
def get_files_data(collection_name:str):
    if collection_name not in get_collection_list():
            raise HTTPException(
                status_code=404, detail="Collection name is incorrect or does not exist"
            )
    data = get_collection_stats(collection_name=collection_name)
    return ingest_data_response(**data)


@router.post("/files/{collection_name}", response_model=ingest_created_response)
async def ingest_files(
    collection_name:str,
    files: List[FileUpload]=File(...),
):
    if collection_name not in get_collection_list():
        raise HTTPException(
            status_code=404, detail="Collection name is incorrect or does not exist"
        )

    loader = SmartLoaderManager()
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths = []
        for upload in files:
            if not upload.filename:
                continue
            safe_name = Path(upload.filename).name
            dest = Path(tmp_dir) / safe_name
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(str(dest))

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No files provided.")

        loader.get_from_file(saved_paths)

    docs_len = len(
        {docs.metadata.get("file_name", "unknown") for docs in loader.docs_loaded}
    )
    chunks = recursive_chunking(loader.docs_loaded)
    response = add_chunks(get_vector_db(collection_name), chunks)

    if response[0] != 200:
        raise HTTPException(status_code=response[0], detail=response[1])

    return ingest_created_response(
        collection_name=collection_name.lower(),
        files_received=docs_len,
        chunks_created=len(chunks),
        ingest_status=response[1],
    )


@router.post("/git", response_model=ingest_created_response)
async def ingest_git(request_data:ingest_request_model):
    if request_data.collection_name not in get_collection_list():
        raise HTTPException(
            status_code=404, detail="Collection name is incorrect or does not exist"
        )

    if not (request_data.git_url.endswith(".git") and request_data.git_url.startswith("http")):
        raise HTTPException(status_code=504, detail="Git URL is not valid")

    loader = SmartLoaderManager()
    git_docs = loader.fetch_git_repo(
        repo_link=request_data.git_url, repo_name=request_data.git_url[:-4].split("/")[-1]
    )

    docs_len = len(
        {docs.metadata.get("file_name", "unknown") for docs in loader.docs_loaded}
    )
    chunks = recursive_chunking(loader.docs_loaded)
    response = add_chunks(get_vector_db(request_data.collection_name), chunks)

    if response[0] != 200:
        raise HTTPException(status_code=response[0], detail=response[1])

    return ingest_created_response(
        collection_name=request_data.collection_name.lower(),
        files_received=docs_len,
        chunks_created=len(chunks),
        ingest_status=response[1],
    )

@router.delete('/delete',response_model=default_response)
def delete_chunks_api(request_data:delete_chunks_request_model):
    if request_data.collection_name not in get_collection_list():
        raise HTTPException(
            status_code=404, detail="Collection name is incorrect or does not exist"
        )
    if get_collection_stats(collection_name=request_data.collection_name,file_name=request_data.file_name).get('total_chunks') <1:
        raise HTTPException(status_code=503,detail="File name is correct or file is not inserted in vector database")
    kwargs = {"collection_name":request_data.collection_name,"file_name":request_data.file_name}
    if request_data.file_type :
        kwargs['file-type']= request_data.file_type

    fcount=delete_chunks(**kwargs)
    if fcount ==0:
        raise HTTPException(status_code=404,detail="Chunks related to file does not exist")
    else:
        return default_response(message=f"total {str(fcount)} chunks deleted ")
