# rag/__init__.py
"""
Dheera v0.3.0 - RAG Package
Retrieval Augmented Generation components.
"""

from rag.embeddings import EmbeddingModel, embed_text, embed_texts, get_embedding_model
from rag.vector_store import VectorStore, VectorDocument
from rag.retriever import RAGRetriever, RetrievalResult

__all__ = [
    "EmbeddingModel",
    "embed_text",
    "embed_texts",
    "get_embedding_model",
    "VectorStore",
    "VectorDocument",
    "RAGRetriever",
    "RetrievalResult",
]
