import asyncio
from fastapi import FastAPI,HTTPException
from config import get_chatllm, get_embeddingmodel
from core import get_vectordb_client
from routes import collection_router,ingestion_router
from config.api_schema import health_response,chat_request_model,chat_response
from core.chain import get_chain,format_history_chat
from langchain_core.messages import BaseMessage
from core.vectordb import get_vector_db,get_collection_list
from typing import List

app = FastAPI(
    title="Langchain RAG",
    description="A RAG System powered by FastAPI",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.include_router(collection_router)
app.include_router(ingestion_router)


@app.get("/api/",tags=['operational'],description="root")
def root():
    """Root endpoint returning API information."""
    return {"message": "LangChain RAG API", "version": "1.0.0"}



async def _check_chatllm(timeout: float = 3.0) -> bool:
    try:
        llm = get_chatllm()
        await asyncio.wait_for(llm.bind(max_tokens=1).ainvoke("ping"), timeout=timeout)
        return True
    except Exception:
        try:
            await asyncio.wait_for(llm.ainvoke("ping"), timeout=timeout)
            return True
        except Exception:
            return False



async def _check_embedding(timeout: float = 1.0) -> bool:
    try:
        embedd = get_embeddingmodel()
        await asyncio.wait_for(embedd.aembed_query("ping"), timeout=timeout)
        return True
    except Exception:
        return False


async def _check_vectordb(timeout: float = 1.0) -> bool:
    try:
        client = get_vectordb_client()
        await asyncio.wait_for(asyncio.to_thread(client.get_collections), timeout=timeout)
        return True
    except Exception:
        return False



@app.get(
    "/api/health",
    description="Check connectivity status of API, LLM, Embedding, and VectorDB services",
    tags=['operational'],
    response_model=health_response,
)
async def health():
    """Checks operational status of major services concurrently."""
    chat_llm, embedding, vectordb = await asyncio.gather(
        _check_chatllm(),
        _check_embedding(),
        _check_vectordb(),
    )

    return health_response(
        api_services=True,
        chat_LLM_services=chat_llm,
        embedding_Model_services=embedding,
        vectordb_services=vectordb,
    )

@app.post('/api/chat',tags=['Chat'],response_model=chat_response)
def chat(request_data:chat_request_model):
    try:
        if request_data.collection_name not in get_collection_list():
            raise HTTPException(status_code=404 , detail="Collection name is incorrect or does not exist")
        history = []
        lang_history:List[BaseMessage]
        kwargs = {"question":request_data.query}
        if request_data.history != None:
            history= request_data.history
            lang_history= [format_history_chat(chat) for chat in history]
            kwargs["history"]= lang_history
        else :
            kwargs["history"]=[]
        db = get_vector_db(request_data.collection_name.lower())
        chain = get_chain(db)
        response=chain.invoke({**kwargs})
        history.extend([('user',request_data.query),('ai',response)])
        return chat_response(airesponse=response,history=history)
    except Exception as E:
        print(E)
        raise HTTPException(status_code=503,detail="Unable to process your query at this time")
