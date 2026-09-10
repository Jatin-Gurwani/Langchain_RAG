from fastapi import APIRouter, HTTPException
from core.vectordb import get_collection_list,get_collection_data_by_name,create_collection,delete_collection,get_collection_stats
from config.api_schema import collections_name_list_response,collection_baserequest,single_collection_response,default_response

router = APIRouter(prefix="/api/collections", tags=['collections'])


@router.get('/', description='provides list of collections created by user', response_model=collections_name_list_response)
def full_collections_list_api():
    try:
        clist = get_collection_list()
        return collections_name_list_response(collections=clist)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch collections: {e}")

@router.get('/{collection_name}',description=" provides vector points for given collection name",response_model=single_collection_response)
def get_collection_data_api(collection_name:str):

        data = get_collection_data_by_name(collection_name.lower())
        print(data)
        stats = get_collection_stats(collection_name)
        chunksc=stats.get('total_chunks')
        filesc = len(stats.get('filenames'))
        if data[0] != 200:
            raise HTTPException(status_code=data[0],detail=data[1])
        return single_collection_response(name=collection_name.lower(),status="active",total_chunks=chunksc,files_count=filesc)

@router.post("/",response_model=default_response, description="creates collection in vector database")
def create_collection_api(request_data:collection_baserequest):
    results= create_collection(request_data.name)
    if results[0] != 200:
        raise HTTPException(status_code=results[0],detail=results[1])
    else:
         return default_response(message=results[1])

@router.delete("/",response_model=default_response,description="deletes collection's from vector database")
def delete_collection_api(request_data:collection_baserequest):
    results= delete_collection(request_data.name)
    if results[0] != 200:
        raise HTTPException(status_code=results[0],detail=results[1])
    else:
         return default_response(message=results[1])    


