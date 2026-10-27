"""Shared chat-model and embedding initializers, so notebooks and apps build them the same way.

Each provider's package is imported only when it is used: HuggingFace pulls in torch and
sentence-transformers, which takes seconds and is not needed when you only call OpenAI.
"""

from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from . import config  # noqa: F401  (importing config loads the API keys from .env)

ChatProvider = Literal["openai", "groq", "ollama"]
EmbeddingProvider = Literal["openai", "huggingface", "ollama"]

DEFAULT_CHAT_MODELS = {
    "openai": "gpt-5",
    "groq": "openai/gpt-oss-120b",
    "ollama": "llama3.2",              # any model you have pulled: `ollama pull llama3.2`
}

DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",   # runs locally, no API key
    "ollama": "nomic-embed-text",
}


def get_llm(
    provider: ChatProvider = "openai",
    model: str | None = None,
    temperature: float = 0,
    **kwargs,
) -> BaseChatModel:
    """A chat model from OpenAI (OPENAI_API_KEY), Groq (GROQ_API_KEY) or a local Ollama server.

    model defaults to DEFAULT_CHAT_MODELS[provider]. Extra kwargs go to the model class
    (max_tokens, timeout, base_url, ...).
    """
    model = model or DEFAULT_CHAT_MODELS.get(provider)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, **kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature, **kwargs)
    raise ValueError(f"Unknown chat provider {provider!r}, expected one of {list(DEFAULT_CHAT_MODELS)}")


def get_embeddings(
    provider: EmbeddingProvider = "openai",
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """An embedding model from OpenAI (OPENAI_API_KEY), HuggingFace (local) or a local Ollama server.

    model defaults to DEFAULT_EMBEDDING_MODELS[provider]. Use the same model to build a vector
    store and to query it. Extra kwargs go to the model class (dimensions, encode_kwargs, ...).
    """
    model = model or DEFAULT_EMBEDDING_MODELS.get(provider)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, **kwargs)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model, **kwargs)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, **kwargs)
    raise ValueError(f"Unknown embedding provider {provider!r}, expected one of {list(DEFAULT_EMBEDDING_MODELS)}")
