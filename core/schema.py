"""
Schema definitions for LangChain RAG project.
Contains type definitions and schema models used across the application.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata schema for RAG documents."""
    file_name: str = Field(description="Name of the source file")
    file_type: str = Field(description="File extension/type")
    file_path: str = Field(default="", description="Path to the source file")
    
    class Config:
        arbitrary_types_allowed = True


class ChunkMetadata(BaseModel):
    """Metadata schema for document chunks."""
    doc_id: Optional[str] = None
    file_name: str = Field(description="Name of the source file")
    file_type: str = Field(description="File extension/type")
    file_path: str = Field(default="", description="Path to the source file")


class VectorStoreSchema(BaseModel):
    """Schema for vector database configurations."""
    name: str
    embedding_size: int
    sparse_vector_name: Optional[str] = None
    dense_vector_name: Optional[str] = None