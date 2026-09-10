from pydantic import BaseModel,WithJsonSchema
from typing import List,Optional,Annotated,Dict,Tuple
from fastapi import Form, File,UploadFile
from langchain_core.messages import BaseMessage


class default_response(BaseModel):
    message: str

class health_response(BaseModel):
    api_services: bool = True
    chat_LLM_services: bool = False
    embedding_Model_services: bool = False
    vectordb_services: bool = False

class collections_name_list_response(BaseModel):
    collections: List[str]

class single_collection_response(BaseModel):
    name:str
    status:str
    #optimizer_status:str
    #indexed_vectors_count:int
    total_chunks:int
    files_count:int

class collection_baserequest(BaseModel):
    name:str

class ingest_request_model(BaseModel):
    collection_name:str
    git_url:str


class ingest_created_response(BaseModel):
    collection_name:str
    files_received: int
    chunks_created:int
    ingest_status:str

FileUpload = Annotated[UploadFile, WithJsonSchema({"type": "string", "format": "binary"})]

class delete_chunks_request_model(BaseModel):
    collection_name:str
    file_name:str
    file_type:Optional[str]=None
    file_id:Optional[str]=None

class ingest_data(BaseModel):
    file_name:str
    chunks:int

class ingest_data_response(BaseModel):
    total_chunks:int
    filenames:List[str]
    files:List[ingest_data]

class chat_request_model(BaseModel):
    collection_name:str
    query:str
    #history:Optional[List[BaseMessage]]=None
    history:Optional[list[Tuple]]=None

class chat_response(BaseModel):
    airesponse: str
    history:List[Tuple]