# rag/vector_store.py
"""
Dheera v0.3.0 - Vector Store
Handles vector storage and retrieval using ChromaDB or fallback.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

from rag.embeddings import EmbeddingModel


@dataclass
class VectorDocument:
    """A document stored in the vector store."""
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    score: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    """
    Vector store for semantic search.
    Uses ChromaDB if available, falls back to numpy-based search.
    """
    
    def __init__(
        self,
        collection_name: str = "dheera_memory",
        persist_directory: str = "./chroma_db",
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or EmbeddingModel()
        
        self._use_chroma = False
        self._chroma_client = None
        self._collection = None
        
        # Fallback storage
        self._documents: Dict[str, VectorDocument] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        
        self._init_store()
    
    def _init_store(self):
        """Initialize the vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB
            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            ))
            
            # Get or create collection
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"Dheera {self.collection_name}"}
            )
            
            self._use_chroma = True
            print(f"✓ ChromaDB initialized: {self.collection_name}")
            
        except ImportError:
            print("⚠ ChromaDB not available, using fallback vector store")
            self._use_chroma = False
        except Exception as e:
            print(f"⚠ ChromaDB error: {e}, using fallback")
            self._use_chroma = False
    
    def add(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """
        Add a document to the store.
        
        Args:
            id: Unique document ID
            text: Document text
            metadata: Optional metadata
            embedding: Pre-computed embedding (computed if not provided)
        """
        metadata = metadata or {}
        metadata["added_at"] = datetime.now().isoformat()
        
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.embedding_model.embed(text)
        
        if self._use_chroma:
            # Filter metadata to only include valid types
            clean_metadata = self._clean_metadata(metadata)
            
            self._collection.add(
                ids=[id],
                embeddings=[embedding.tolist()],
                documents=[text],
                metadatas=[clean_metadata],
            )
        else:
            # Fallback storage
            self._documents[id] = VectorDocument(
                id=id,
                text=text,
                embedding=embedding,
                metadata=metadata,
            )
            self._embeddings[id] = embedding
    
    def add_batch(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        """Add multiple documents at once."""
        if not ids:
            return
        
        metadatas = metadatas or [{} for _ in ids]
        
        # Add timestamp to all metadata
        for meta in metadatas:
            meta["added_at"] = datetime.now().isoformat()
        
        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(texts)
        
        if self._use_chroma:
            clean_metadatas = [self._clean_metadata(m) for m in metadatas]
            
            self._collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=clean_metadatas,
            )
        else:
            for i, (id_, text, meta) in enumerate(zip(ids, texts, metadatas)):
                self._documents[id_] = VectorDocument(
                    id=id_,
                    text=text,
                    embedding=embeddings[i],
                    metadata=meta,
                )
                self._embeddings[id_] = embeddings[i]
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of VectorDocument sorted by relevance
        """
        query_embedding = self.embedding_model.embed(query)
        return self.search_by_embedding(query_embedding, n_results, filter_metadata)
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        """Search using a pre-computed embedding."""
        
        if self._use_chroma:
            # ChromaDB search
            where_filter = filter_metadata if filter_metadata else None
            
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter,
            )
            
            documents = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc = VectorDocument(
                        id=results['ids'][0][i],
                        text=results['documents'][0][i] if results['documents'] else "",
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                        score=1.0 - results['distances'][0][i] if results['distances'] else 0.0,
                    )
                    documents.append(doc)
            
            return documents
        
        else:
            # Fallback numpy search
            if not self._embeddings:
                return []
            
            # Stack all embeddings
            ids = list(self._embeddings.keys())
            embeddings = np.array([self._embeddings[id_] for id_ in ids])
            
            # Compute similarities
            similarities = self.embedding_model.batch_similarity(query_embedding, embeddings)
            
            # Sort by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Filter and return top results
            documents = []
            for idx in sorted_indices[:n_results]:
                id_ = ids[idx]
                doc = self._documents[id_]
                
                # Apply metadata filter
                if filter_metadata:
                    match = all(
                        doc.metadata.get(k) == v
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue
                
                doc.score = float(similarities[idx])
                documents.append(doc)
            
            return documents
    
    def get(self, id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        if self._use_chroma:
            results = self._collection.get(ids=[id])
            if results['ids']:
                return VectorDocument(
                    id=results['ids'][0],
                    text=results['documents'][0] if results['documents'] else "",
                    metadata=results['metadatas'][0] if results['metadatas'] else {},
                )
            return None
        else:
            return self._documents.get(id)
    
    def delete(self, id: str):
        """Delete a document by ID."""
        if self._use_chroma:
            self._collection.delete(ids=[id])
        else:
            if id in self._documents:
                del self._documents[id]
            if id in self._embeddings:
                del self._embeddings[id]
    
    def delete_batch(self, ids: List[str]):
        """Delete multiple documents."""
        if self._use_chroma:
            self._collection.delete(ids=ids)
        else:
            for id_ in ids:
                self.delete(id_)
    
    def count(self) -> int:
        """Get total document count."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._documents)
    
    def clear(self):
        """Clear all documents."""
        if self._use_chroma:
            # Recreate collection
            self._chroma_client.delete_collection(self.collection_name)
            self._collection = self._chroma_client.create_collection(
                name=self.collection_name,
            )
        else:
            self._documents.clear()
            self._embeddings.clear()
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for ChromaDB (only allows str, int, float, bool)."""
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean
    
    def persist(self):
        """Persist to disk (for ChromaDB)."""
        if self._use_chroma and self._chroma_client:
            self._chroma_client.persist()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "backend": "chromadb" if self._use_chroma else "numpy",
            "collection": self.collection_name,
            "document_count": self.count(),
            "embedding_dim": self.embedding_model.embedding_dim,
        }


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing VectorStore...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=tmpdir,
        )
        
        print(f"✓ Backend: {store.get_stats()['backend']}")
        
        # Add documents
        store.add(
            id="doc1",
            text="Python is a great programming language for machine learning",
            metadata={"category": "programming"},
        )
        
        store.add(
            id="doc2",
            text="TensorFlow and PyTorch are popular deep learning frameworks",
            metadata={"category": "ml"},
        )
        
        store.add(
            id="doc3",
            text="The weather is sunny today",
            metadata={"category": "weather"},
        )
        
        print(f"✓ Added {store.count()} documents")
        
        # Search
        results = store.search("machine learning programming", n_results=2)
        print(f"✓ Search results:")
        for doc in results:
            print(f"  - {doc.id}: {doc.text[:50]}... (score: {doc.score:.3f})")
        
        # Get by ID
        doc = store.get("doc1")
        print(f"✓ Get by ID: {doc.text[:50]}...")
        
        # Delete
        store.delete("doc3")
        print(f"✓ After delete: {store.count()} documents")
        
        print(f"✓ Stats: {store.get_stats()}")
        
    print("\n✅ Vector store tests passed!")
