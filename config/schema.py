from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from os import getenv,path
from dotenv import load_dotenv
from pathlib import Path

# Valid providers for each type
CHAT_LLM_PROVIDERS = Literal["ollama", "google", "openai", "lmstudio"]
EMBEDDING_PROVIDERS = Literal["ollama", "google", "openai","lmstudio"]

VALID_CHAT_PROVIDERS: set = {"ollama", "google", "openai", "lmstudio"}
VALID_EMBEDDING_PROVIDERS: set = {"ollama", "google", "openai","lmstudio"}


class Settings(BaseSettings):
    """Configuration settings for RAG models and vector stores."""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Chat LLM Configuration
    chatllm_base_url: str = "http://localhost:11434"
    chatllm_provider: CHAT_LLM_PROVIDERS = "ollama"
    chatllm_model: str = "gemma4:e4b-it-qat"
    chatllm_condense_model: str | None = "gemma4:e4b-it-qat"  # Optional cheap/fast model for query rewriting
    chatllm_request_timeout: int = 360

    # Embedding Model Configuration
    embedding_base_url: str = "http://localhost:11434"
    embedding_provider: EMBEDDING_PROVIDERS = "ollama"
    embedding_model: str = "embeddinggemma:latest"
    embedding_size: int = 768
    embedding_request_timeout: int = 120

    # Vision LLM Configuration (if needed)
    visionllm_provider: CHAT_LLM_PROVIDERS = "ollama"
    visionllm_base_url: str = "http://localhost:11434"
    vision_model: str = "gemma4:e4b-it-qat"

    # Vector Database Configuration
    vector_db_provider: Literal['qdrant','chroma'] ='qdrant'
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    sparse_embedding_model: str | None = "Qdrant/bm25"  # Optional for hybrid search
    fastembed_cache_dir: str | None = "/app/.cache/fastembed"

    # Local storage paths
    local_path: str = "./"
    chroma_db_path: str = "./chroma_vector_db/"
    local_repo_path: str = "./repo/"

    # Filter metadata fields (if used with hybrid search)
    matadata_filter_allowed: list[str] | None = ['file_type', 'file_name']


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance to avoid reloading from .env on each call.
    
    Returns:
        Settings: Configuration object with model and service settings.
    """
    load_dotenv()
    return Settings()


@lru_cache(maxsize=1)
def get_chatllm() -> ChatOpenAI | ChatOllama | ChatGoogleGenerativeAI:
    """Get a cached chat LLM instance based on configured provider.
    
    Supported providers:
        - ollama: Uses Ollama API (default, requires BASE_URL to be set)
        - openai: Uses OpenAI-compatible API (requires OPENAI_API_KEY env var)
        - google: Uses Google Generative AI (requires GOOGLE_API_KEY env var)
        - lmstudio: Uses LM Studio via OpenAI-compatible API
    
    Returns:
        ChatOpenAI, ChatOllama, or ChatGoogleGenerativeAI instance ready for use.
    
    Raises:
        ValueError: If chatllm_provider is invalid or missing required environment variables.
        
    Example:
        >>> from config.schema import get_chatllm
        >>> llm = get_chatllm()
        >>> response = llm.invoke("What is RAG?")
    """
    settings = get_settings()
    
    # Validate provider
    if settings.chatllm_provider not in VALID_CHAT_PROVIDERS:
        raise ValueError(
            f"Invalid chat LLM provider '{settings.chatllm_provider}'. "
            f"Valid options are: {', '.join(sorted(VALID_CHAT_PROVIDERS))}"
        )
    
    # Map provider to appropriate LangChain class
    llm_classes = {
        "ollama": ChatOllama,
        "openai": ChatOpenAI,
        "google": ChatGoogleGenerativeAI,
        "lmstudio": ChatOpenAI,  # LM Studio uses OpenAI API format
    }
    
    api_keys = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": None,  # Ollama typically doesn't require API key
        "lmstudio": None,  # LM Studio uses local instance
    }
    
    llm_class = llm_classes[settings.chatllm_provider]
    api_key_env = api_keys[settings.chatllm_provider]
    
    # Get and validate API key for providers that require it
    api_key = "NONE"
    if api_key_env:
        api_key = str(getenv(api_key_env, "")).strip()
        if not api_key:
            raise ValueError(
                f"Missing required environment variable '{api_key_env}'. "
                f"Please set it in your .env file to use the '{settings.chatllm_provider}' provider."
            )


    # Build instance with appropriate parameters per provider
    kwargs = {
        "model": settings.chatllm_model,
        "timeout": settings.chatllm_request_timeout,
        "api_key":api_key
    }
    
    # Add base_url for Ollama and LM Studio (which use local endpoints)
    if settings.chatllm_provider in ("ollama", "lmstudio"):
        kwargs["base_url"] = settings.chatllm_base_url
    
    # Add temperature/deterministic settings (optional defaults)
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.2  # Default creative temperature
    
    return llm_class(**kwargs)


@lru_cache(maxsize=1)
def get_embeddingmodel() -> OpenAIEmbeddings | OllamaEmbeddings | GoogleGenerativeAIEmbeddings:
    """Get a cached embedding model instance based on configured provider.
    
    Supported providers:
        - ollama: Uses Ollama API (default, requires BASE_URL to be set)
        - openai: Uses OpenAI-compatible API (requires OPENAI_API_KEY env var)
        - google: Uses Google Generative AI Embeddings (requires GOOGLE_API_KEY env var)
    
    Returns:
        OpenAIEmbeddings, OllamaEmbeddings, or GoogleGenerativeAIEmbeddings instance.
    
    Raises:
        ValueError: If embedding_provider is invalid or missing required environment variables.
        
    Example:
        >>> from config.schema import get_embeddingmodel
        >>> embeddings = get_embeddingmodel()
        >>> vector = embeddings.embed_query("What is RAG?")
    """
    settings = get_settings()
    
    # Validate provider
    if settings.embedding_provider not in VALID_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Invalid embedding provider '{settings.embedding_provider}'. "
            f"Valid options are: {', '.join(sorted(VALID_EMBEDDING_PROVIDERS))}"
        )
    
    # Map provider to appropriate LangChain class
    embedding_classes = {
        "ollama": OllamaEmbeddings,
        "openai": OpenAIEmbeddings,
        "google": GoogleGenerativeAIEmbeddings,
        "lmstudio": OpenAIEmbeddings
    }
    
    api_keys = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": None,
        "lmstudio": None    # Ollama/LM Studio typically doesn't require API key
    }
    
    embedding_class = embedding_classes[settings.embedding_provider]
    api_key_env = api_keys[settings.embedding_provider]
    
    # Get and validate API key for providers that require it
    api_key = "NONE"
    if api_key_env:
        api_key = str(getenv(api_key_env, "")).strip()
        if not api_key:
            raise ValueError(
                f"Missing required environment variable '{api_key_env}'. "
                f"Please set it in your .env file to use the '{settings.embedding_provider}' provider."
            )

    
    # Build instance with appropriate parameters per provider
    kwargs = {
        "model": settings.embedding_model,
        "timeout": settings.embedding_request_timeout,
        "check_embedding_ctx_length": False,
        "api_key":api_key  # Disable context length checking for performance
    }
    
    # Add base_url for Ollama
    if settings.embedding_provider in ["ollama",'lmstudio']:
        kwargs["base_url"] = settings.embedding_base_url
    
    return embedding_class(**kwargs)

