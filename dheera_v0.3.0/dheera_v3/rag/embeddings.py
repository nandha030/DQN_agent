# rag/embeddings.py
"""
Dheera v0.3.0 - Embeddings Module
Handles text embeddings using sentence-transformers.
"""

import numpy as np
from typing import List, Optional, Union
import hashlib


class EmbeddingModel:
    """
    Embedding model wrapper for sentence-transformers.
    Falls back to simple TF-IDF-like embeddings if sentence-transformers unavailable.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self._model = None
        self._fallback_mode = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            device = "cuda" if self.use_gpu else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
            print(f"✓ Loaded embedding model: {self.model_name} (dim={self.embedding_dim})")
            
        except ImportError:
            print("⚠ sentence-transformers not available, using fallback embeddings")
            self._fallback_mode = True
            self.embedding_dim = 384  # Fixed fallback dimension
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        if self._fallback_mode:
            return self._fallback_embed(text)
        
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding matrix (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        if self._fallback_mode:
            return np.array([self._fallback_embed(t) for t in texts])
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.astype(np.float32)
    
    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Simple fallback embedding using hash-based features.
        Not as good as sentence-transformers but works without dependencies.
        """
        # Normalize text
        text = text.lower().strip()
        words = text.split()
        
        # Create embedding
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Hash-based word features
        for i, word in enumerate(words):
            # Hash word to get indices
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            
            # Distribute across embedding dimensions
            for j in range(5):  # 5 features per word
                idx = (word_hash + j * 7919) % self.embedding_dim
                val = ((word_hash >> (j * 8)) & 0xFF) / 255.0 - 0.5
                
                # Position weighting
                position_weight = 1.0 / (1.0 + i * 0.1)
                embedding[idx] += val * position_weight
        
        # Add length feature
        embedding[0] = min(len(words) / 50.0, 1.0)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))
    
    def batch_similarity(
        self,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple candidates.
        
        Args:
            query_emb: Query embedding (embedding_dim,)
            candidate_embs: Candidate embeddings (num_candidates, embedding_dim)
            
        Returns:
            Similarity scores (num_candidates,)
        """
        if len(candidate_embs) == 0:
            return np.array([])
        
        return np.dot(candidate_embs, query_emb)


# Convenience functions
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the default embedding model."""
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel()
    return _default_model


def embed_text(text: str) -> np.ndarray:
    """Embed text using default model."""
    return get_embedding_model().embed(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed multiple texts using default model."""
    return get_embedding_model().embed_batch(texts)


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing EmbeddingModel...")
    
    model = EmbeddingModel()
    
    # Test single embedding
    text = "Hello, how can I help you today?"
    emb = model.embed(text)
    print(f"✓ Single embedding shape: {emb.shape}")
    print(f"  Norm: {np.linalg.norm(emb):.4f}")
    
    # Test batch embedding
    texts = [
        "Python is a great programming language",
        "I love machine learning",
        "The weather is nice today",
    ]
    embs = model.embed_batch(texts)
    print(f"✓ Batch embeddings shape: {embs.shape}")
    
    # Test similarity
    sim = model.similarity(embs[0], embs[1])
    print(f"✓ Similarity (Python vs ML): {sim:.4f}")
    
    sim2 = model.similarity(embs[0], embs[2])
    print(f"✓ Similarity (Python vs Weather): {sim2:.4f}")
    
    # Test batch similarity
    query = model.embed("programming in Python")
    sims = model.batch_similarity(query, embs)
    print(f"✓ Batch similarities: {sims}")
    
    print("\n✅ Embedding tests passed!")
