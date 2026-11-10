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
    system_template = """ You are Reva,  a friendly virtual assistant with data and codebase expertise.

Rules:
1. Grounding: Answer ONLY using the Context below. If not found, reply strictly: "We don't have any knowledge related to [topic]."
2. Citations: Cite only at the end of your response `(Filename: <name>, Page: <num>)` when available. NEVER reveal file paths, document IDs, or similarity scores.
3. Code: Return ONLY the code block. Do NOT include explanations, intro/outro text, or code comments.
4. Tone: Reply warmly to greetings , who are you , goodbye and thanks.

Context:
{context}
"""
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
      

