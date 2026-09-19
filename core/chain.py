from langchain import agents
from langchain_core.runnables import RunnablePassthrough,RunnableBranch,RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.documents import Document
from core.vectordb import retrieve_chunks, get_vector_db
from langchain_qdrant import QdrantVectorStore
from langchain_chroma import Chroma
from typing import List,Tuple
from config import get_chatllm,get_embeddingmodel




def get_chain(db:QdrantVectorStore|Chroma):
    llm = get_chatllm()
    system_template = """ You are frendly personal assistant who has experience in understanding data and codebase.
Answer the question using ONLY the context provided below.
If the answer cannot be found in the context, say "you don't have any knowledge related to ..." except greeting and who are you queries.
Provide specific references of filename , page number if available . donot share document id ,file path or score at any cost.
if the user ask for code snippet, provide the code snippet only and do not provide any explanation or comments.

Context:
{context} """
    prompt = ChatPromptTemplate.from_messages([("system",system_template),MessagesPlaceholder('history'),("human","{question}")])
    
    contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system","Given the chat history and the latest question, rewrite the question "
    "as a fully standalone question. Do NOT answer — only rewrite. "
    "If already standalone, return as-is."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
    ])
    contextualize_chain = RunnableBranch(
    (
        lambda x: len(x["history"]) > 0,
        contextualize_prompt | llm | StrOutputParser(),
    ),
    lambda x: x["question"],   
)

    chain = (
    RunnableParallel({                                    
        "standalone": contextualize_chain,
        "question":   lambda x: x["question"],
        "history":    lambda x: x["history"],
    })
    | RunnableParallel({                                  
        "context":  lambda x: retrieve_chunks(db=db,query=x["standalone"]),
        "question": lambda x: x["question"],
        "history":  lambda x: x["history"],
    })
    | prompt
    | llm
    | StrOutputParser()
    )
    return chain

def format_history(current_history:Tuple):
        message = []
        human_chat,ai_chat = current_history
        message.append(HumanMessage(content=human_chat))
        message.append(AIMessage(content=ai_chat))
        return message

def format_history_chat(chat:Tuple):
      role_dict={'ai':AIMessage,'user':HumanMessage}
      role,content = chat
      return role_dict[role](content=content)
      

