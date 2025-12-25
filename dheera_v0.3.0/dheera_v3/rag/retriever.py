# rag/retriever.py
"""
Dheera v0.3.0 - RAG Retriever
Retrieval-Augmented Generation system for context retrieval.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import json

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore, VectorDocument


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    documents: List[VectorDocument]
    query: str
    query_embedding: np.ndarray
    total_found: int
    retrieval_time_ms: float
    
    def get_context_string(self, max_docs: int = 5, max_chars: int = 2000) -> str:
        """Format retrieved documents as context string for SLM."""
        parts = []
        total_chars = 0
        
        for doc in self.documents[:max_docs]:
            text = doc.text
            if total_chars + len(text) > max_chars:
                text = text[:max_chars - total_chars] + "..."
            
            parts.append(f"[{doc.metadata.get('source', 'memory')}] {text}")
            total_chars += len(text)
            
            if total_chars >= max_chars:
                break
        
        return "\n---\n".join(parts)


class RAGRetriever:
    """
    RAG System for Dheera.
    
    Manages multiple collections:
    - conversations: Past conversation turns
    - knowledge: External knowledge/documents  
    - search_results: Cached web search results
    
    Provides unified retrieval across all sources.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or EmbeddingModel()
        
        # Initialize collections
        self.conversations = VectorStore(
            collection_name="conversations",
            persist_directory=persist_directory,
            embedding_model=self.embedding_model,
        )
        
        self.knowledge = VectorStore(
            collection_name="knowledge",
            persist_directory=persist_directory,
            embedding_model=self.embedding_model,
        )
        
        self.search_cache = VectorStore(
            collection_name="search_cache",
            persist_directory=persist_directory,
            embedding_model=self.embedding_model,
        )
        
        # Statistics
        self._retrieval_count = 0
        self._total_retrieval_time = 0.0
    
    # ==================== Add Methods ====================
    
    def add_conversation_turn(
        self,
        turn_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a conversation turn to memory.
        
        Args:
            turn_id: Unique turn identifier
            user_message: What the user said
            assistant_response: What the assistant replied
            metadata: Additional metadata (action, reward, etc.)
        """
        # Combine for embedding
        combined_text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        metadata = metadata or {}
        metadata.update({
            "source": "conversation",
            "user_message": user_message[:200],  # Truncate for metadata
            "turn_id": turn_id,
        })
        
        self.conversations.add(
            id=turn_id,
            text=combined_text,
            metadata=metadata,
        )
    
    def add_knowledge(
        self,
        content: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add external knowledge to the system.
        
        Args:
            content: Knowledge content
            source: Source identifier (e.g., "manual", "document", "wiki")
            doc_id: Optional document ID
            metadata: Additional metadata
        """
        doc_id = doc_id or str(uuid.uuid4())
        
        metadata = metadata or {}
        metadata.update({
            "source": source,
            "content_type": "knowledge",
        })
        
        self.knowledge.add(
            id=doc_id,
            text=content,
            metadata=metadata,
        )
    
    def add_search_result(
        self,
        query: str,
        result_text: str,
        url: Optional[str] = None,
        result_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a web search result to cache.
        
        Args:
            query: Original search query
            result_text: Search result content
            url: Source URL
            result_id: Optional result ID
            metadata: Additional metadata
        """
        result_id = result_id or str(uuid.uuid4())
        
        metadata = metadata or {}
        metadata.update({
            "source": "web_search",
            "query": query[:100],
            "url": url or "",
        })
        
        self.search_cache.add(
            id=result_id,
            text=result_text,
            metadata=metadata,
        )
    
    # ==================== Retrieve Methods ====================
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        sources: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """
        Retrieve relevant context from all sources.
        
        Args:
            query: Search query
            n_results: Max results to return
            sources: Which sources to search (None = all)
            min_score: Minimum similarity score threshold
            
        Returns:
            RetrievalResult with documents
        """
        import time
        start_time = time.time()
        
        sources = sources or ["conversations", "knowledge", "search_cache"]
        query_embedding = self.embedding_model.embed(query)
        
        all_documents = []
        
        # Search each source
        if "conversations" in sources:
            conv_results = self.conversations.search_by_embedding(
                query_embedding, n_results=n_results
            )
            all_documents.extend(conv_results)
        
        if "knowledge" in sources:
            knowledge_results = self.knowledge.search_by_embedding(
                query_embedding, n_results=n_results
            )
            all_documents.extend(knowledge_results)
        
        if "search_cache" in sources:
            search_results = self.search_cache.search_by_embedding(
                query_embedding, n_results=n_results
            )
            all_documents.extend(search_results)
        
        # Filter by score
        if min_score > 0:
            all_documents = [d for d in all_documents if d.score >= min_score]
        
        # Sort by score and limit
        all_documents.sort(key=lambda x: x.score, reverse=True)
        all_documents = all_documents[:n_results]
        
        # Track stats
        elapsed_ms = (time.time() - start_time) * 1000
        self._retrieval_count += 1
        self._total_retrieval_time += elapsed_ms
        
        return RetrievalResult(
            documents=all_documents,
            query=query,
            query_embedding=query_embedding,
            total_found=len(all_documents),
            retrieval_time_ms=elapsed_ms,
        )
    
    def retrieve_conversations(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[VectorDocument]:
        """Retrieve similar past conversations."""
        return self.conversations.search(query, n_results=n_results)
    
    def retrieve_knowledge(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[VectorDocument]:
        """Retrieve relevant knowledge."""
        return self.knowledge.search(query, n_results=n_results)
    
    def retrieve_search_results(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[VectorDocument]:
        """Retrieve cached search results."""
        return self.search_cache.search(query, n_results=n_results)
    
    # ==================== Context Building ====================
    
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 500,
        include_conversations: bool = True,
        include_knowledge: bool = True,
        include_search: bool = True,
    ) -> Tuple[str, List[VectorDocument]]:
        """
        Get formatted context for SLM prompt.
        
        Args:
            query: Current user query
            max_tokens: Approximate max tokens for context
            include_*: Which sources to include
            
        Returns:
            (context_string, documents_used)
        """
        sources = []
        if include_conversations:
            sources.append("conversations")
        if include_knowledge:
            sources.append("knowledge")
        if include_search:
            sources.append("search_cache")
        
        result = self.retrieve(query, n_results=5, sources=sources)
        
        if not result.documents:
            return "", []
        
        # Format context
        # Rough token estimate: 4 chars per token
        max_chars = max_tokens * 4
        context = result.get_context_string(max_docs=5, max_chars=max_chars)
        
        return context, result.documents
    
    def build_augmented_prompt(
        self,
        user_message: str,
        system_prompt: str,
        max_context_tokens: int = 500,
    ) -> str:
        """
        Build a RAG-augmented prompt for SLM.
        
        Args:
            user_message: Current user message
            system_prompt: Base system prompt
            max_context_tokens: Max tokens for retrieved context
            
        Returns:
            Augmented system prompt
        """
        context, docs = self.get_context_for_query(
            user_message,
            max_tokens=max_context_tokens,
        )
        
        if context:
            augmented_prompt = f"""{system_prompt}

RELEVANT CONTEXT (from memory):
{context}

Use the above context to inform your response if relevant."""
            return augmented_prompt
        
        return system_prompt
    
    # ==================== Maintenance ====================
    
    def cleanup_old_entries(
        self,
        collection: str,
        max_age_days: int = 30,
    ) -> int:
        """Remove old entries from a collection."""
        # This would need timestamp filtering support
        # For now, just return 0
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "conversations_count": self.conversations.count(),
            "knowledge_count": self.knowledge.count(),
            "search_cache_count": self.search_cache.count(),
            "total_documents": (
                self.conversations.count() +
                self.knowledge.count() +
                self.search_cache.count()
            ),
            "retrieval_count": self._retrieval_count,
            "avg_retrieval_ms": (
                self._total_retrieval_time / max(self._retrieval_count, 1)
            ),
            "embedding_dim": self.embedding_model.embedding_dim,
        }
    
    def persist(self):
        """Persist all collections to disk."""
        self.conversations.persist()
        self.knowledge.persist()
        self.search_cache.persist()


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing RAGRetriever...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rag = RAGRetriever(persist_directory=tmpdir)
        
        # Add conversation turns
        rag.add_conversation_turn(
            turn_id="turn_1",
            user_message="How do I read a CSV file in Python?",
            assistant_response="You can use pandas: import pandas as pd; df = pd.read_csv('file.csv')",
            metadata={"action": 0, "reward": 1.0},
        )
        
        rag.add_conversation_turn(
            turn_id="turn_2",
            user_message="What about JSON files?",
            assistant_response="Use the json module: import json; data = json.load(open('file.json'))",
            metadata={"action": 0, "reward": 0.5},
        )
        
        # Add knowledge
        rag.add_knowledge(
            content="Pandas is a fast, powerful, flexible open source data analysis library for Python.",
            source="documentation",
        )
        
        rag.add_knowledge(
            content="TensorFlow is an end-to-end open source platform for machine learning.",
            source="documentation",
        )
        
        # Add search results
        rag.add_search_result(
            query="python csv tutorial",
            result_text="CSV files can be read using Python's built-in csv module or pandas library.",
            url="https://example.com/csv",
        )
        
        print(f"✓ Stats: {rag.get_stats()}")
        
        # Test retrieval
        result = rag.retrieve("How to read data files in Python?")
        print(f"\n✓ Retrieved {result.total_found} documents in {result.retrieval_time_ms:.2f}ms")
        
        for doc in result.documents:
            source = doc.metadata.get('source', 'unknown')
            print(f"  - [{source}] {doc.text[:60]}... (score: {doc.score:.3f})")
        
        # Test context building
        context, docs = rag.get_context_for_query("Help me with pandas")
        print(f"\n✓ Context for SLM:\n{context[:200]}...")
        
        # Test augmented prompt
        prompt = rag.build_augmented_prompt(
            "How do I filter a dataframe?",
            "You are Dheera, a helpful AI assistant.",
        )
        print(f"\n✓ Augmented prompt length: {len(prompt)} chars")
        
    print("\n✅ RAG retriever tests passed!")
