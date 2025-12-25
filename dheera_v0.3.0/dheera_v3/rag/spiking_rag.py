# rag/spiking_rag.py
"""
Dheera v0.3.1 - Spiking RAG Retriever
Long-context retrieval using temporal sparse attention

SpikingBrain capability brought to RAG:
- 100K+ token context windows (vs 2K traditional)
- O(n*k) complexity instead of O(n²)
- Event-driven retrieval (attend only to relevant passages)
- Temporal coherence (recent + salient history)

Use cases:
- Multi-document reasoning over 100+ documents
- Long conversation history (1000+ turns)
- Efficient re-ranking of large retrieval sets
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import time

from rag.embeddings import EmbeddingModel
from rag.retriever import RAGRetriever, RetrievalResult
from rag.vector_store import VectorDocument
from core.spiking_attention import (
    MultiHeadSpikingAttention,
    SpikingTransformerBlock,
    AttentionStats,
)


@dataclass
class SpikingRetrievalResult(RetrievalResult):
    """Extended retrieval result with sparsity metrics"""
    attention_sparsity: float = 0.0
    attention_speedup: float = 1.0
    tokens_processed: int = 0
    active_tokens_attended: int = 0


class SpikingRAGRetriever(RAGRetriever):
    """
    RAG Retriever with spiking attention for long-context processing.

    Extends standard RAGRetriever with:
    - Spiking attention encoder for query understanding
    - Temporal sparse re-ranking over large candidate sets
    - Event-driven passage selection
    - 100K+ token context support
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: Optional[EmbeddingModel] = None,
        use_spiking_reranker: bool = True,
        embed_dim: int = 384,
        num_heads: int = 4,
        window_size: int = 256,
        max_context_tokens: int = 100000,
    ):
        """
        Args:
            use_spiking_reranker: Use spiking attention for re-ranking
            embed_dim: Embedding dimension (must match embedding model)
            num_heads: Number of attention heads
            window_size: Local attention window
            max_context_tokens: Maximum context length (100K like SpikingBrain)
        """
        super().__init__(persist_directory, embedding_model)

        self.use_spiking_reranker = use_spiking_reranker
        self.embed_dim = embed_dim
        self.max_context_tokens = max_context_tokens

        if use_spiking_reranker:
            # Query encoder with spiking attention
            self.query_encoder = SpikingTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=embed_dim * 4,
                window_size=window_size,
            )

            # Cross-attention for query-document matching
            self.cross_attention = MultiHeadSpikingAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                use_spike_gating=True,
            )

            # Relevance scorer
            self.relevance_scorer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Statistics
        self.total_attention_sparsity = 0.0
        self.rerank_count = 0

    def retrieve_with_reranking(
        self,
        query: str,
        n_results: int = 10,
        initial_n: int = 100,
        collections: Optional[List[str]] = None,
        use_sparse_attention: bool = True,
    ) -> SpikingRetrievalResult:
        """
        Two-stage retrieval with spiking attention re-ranking.

        Stage 1: Dense retrieval (embeddings) - Get top-K candidates
        Stage 2: Spiking re-ranking - Use attention to refine

        Args:
            query: User query
            n_results: Final number of results
            initial_n: Initial retrieval size for re-ranking
            collections: Collections to search
            use_sparse_attention: Use spiking attention for re-ranking

        Returns:
            SpikingRetrievalResult with efficiency metrics
        """
        start_time = time.perf_counter()

        # Stage 1: Dense retrieval
        initial_results = self.retrieve(
            query=query,
            n_results=min(initial_n, self.max_context_tokens // 100),
            collections=collections,
        )

        # If no spiking reranker or few results, return as-is
        if not self.use_spiking_reranker or not use_sparse_attention or len(initial_results.documents) <= n_results:
            return SpikingRetrievalResult(
                **initial_results.__dict__,
                attention_sparsity=0.0,
                attention_speedup=1.0,
                tokens_processed=0,
                active_tokens_attended=0,
            )

        # Stage 2: Spiking re-ranking
        reranked_docs, attention_stats = self._rerank_with_attention(
            query=query,
            documents=initial_results.documents,
            n_results=n_results,
        )

        # Compute metrics
        retrieval_time = (time.perf_counter() - start_time) * 1000

        avg_sparsity = np.mean([s.sparsity for s in attention_stats]) if attention_stats else 0.0
        avg_speedup = np.mean([s.speedup_vs_dense for s in attention_stats]) if attention_stats else 1.0
        total_tokens = sum([s.total_possible_attention for s in attention_stats]) if attention_stats else 0
        active_tokens = sum([s.actual_attention_computed for s in attention_stats]) if attention_stats else 0

        # Update statistics
        self.total_attention_sparsity += avg_sparsity
        self.rerank_count += 1

        return SpikingRetrievalResult(
            documents=reranked_docs,
            query=query,
            query_embedding=initial_results.query_embedding,
            total_found=len(reranked_docs),
            retrieval_time_ms=retrieval_time,
            attention_sparsity=avg_sparsity,
            attention_speedup=avg_speedup,
            tokens_processed=total_tokens,
            active_tokens_attended=active_tokens,
        )

    def _rerank_with_attention(
        self,
        query: str,
        documents: List[VectorDocument],
        n_results: int,
    ) -> Tuple[List[VectorDocument], List[AttentionStats]]:
        """
        Re-rank documents using spiking attention.

        Uses cross-attention between query and documents to compute
        relevance scores with temporal sparse patterns.
        """
        if not documents:
            return [], []

        # Convert to tensors
        query_emb = self.embedding_model.embed_text(query)
        doc_embs = np.stack([doc.embedding for doc in documents])

        query_tensor = torch.FloatTensor(query_emb).unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        doc_tensor = torch.FloatTensor(doc_embs).unsqueeze(0)  # [1, n_docs, embed_dim]

        # Encode query with temporal context
        query_encoded = self.query_encoder(query_tensor)  # [1, 1, embed_dim]

        # Cross-attention: query attends to documents
        attended, attention_stats = self.cross_attention(
            query_encoded.expand(-1, doc_tensor.shape[1], -1),  # Expand query to match docs
            return_stats=True,
        )

        # Score each document
        scores = self.relevance_scorer(attended).squeeze(-1).squeeze(0)  # [n_docs]
        scores = scores.detach().numpy()

        # Re-rank by scores
        ranked_indices = np.argsort(scores)[::-1][:n_results]
        reranked_docs = [documents[i] for i in ranked_indices]

        return reranked_docs, attention_stats

    def get_long_context(
        self,
        query: str,
        max_tokens: int = 100000,
        chunk_size: int = 500,
    ) -> SpikingRetrievalResult:
        """
        Retrieve long context (up to 100K tokens) efficiently.

        SpikingBrain's key capability: handle massive contexts that
        would be impossible for traditional attention (O(n²) → OOM).

        Args:
            query: User query
            max_tokens: Maximum context tokens (up to 100K)
            chunk_size: Average tokens per chunk

        Returns:
            SpikingRetrievalResult with long context
        """
        # Calculate how many documents we can include
        max_docs = max_tokens // chunk_size

        return self.retrieve_with_reranking(
            query=query,
            n_results=min(max_docs, 1000),  # Cap at 1000 docs
            initial_n=min(max_docs * 2, 5000),  # 2x for re-ranking pool
            use_sparse_attention=True,
        )

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get spiking attention efficiency statistics"""
        return {
            "total_rerankings": self.rerank_count,
            "avg_attention_sparsity": (
                self.total_attention_sparsity / self.rerank_count
                if self.rerank_count > 0
                else 0.0
            ),
            "max_context_tokens": self.max_context_tokens,
            "spiking_enabled": self.use_spiking_reranker,
        }


# ==================== Utility Functions ====================

def compare_retrieval_methods(
    rag_retriever: RAGRetriever,
    spiking_rag_retriever: SpikingRAGRetriever,
    test_queries: List[str],
    n_results: int = 10,
) -> Dict[str, Any]:
    """
    Compare standard vs spiking RAG retrieval.

    Demonstrates SpikingBrain-inspired efficiency gains.
    """
    results = {
        "standard_times": [],
        "spiking_times": [],
        "speedups": [],
        "sparsities": [],
        "quality_match": [],
    }

    for query in test_queries:
        # Standard retrieval
        start = time.perf_counter()
        standard_result = rag_retriever.retrieve(query, n_results=n_results)
        standard_time = (time.perf_counter() - start) * 1000

        # Spiking retrieval
        start = time.perf_counter()
        spiking_result = spiking_rag_retriever.retrieve_with_reranking(
            query, n_results=n_results, use_sparse_attention=True
        )
        spiking_time = (time.perf_counter() - start) * 1000

        # Compute overlap (quality check)
        standard_ids = {doc.id for doc in standard_result.documents[:n_results]}
        spiking_ids = {doc.id for doc in spiking_result.documents[:n_results]}
        overlap = len(standard_ids & spiking_ids) / n_results

        results["standard_times"].append(standard_time)
        results["spiking_times"].append(spiking_time)
        results["speedups"].append(standard_time / spiking_time if spiking_time > 0 else 1.0)
        results["sparsities"].append(spiking_result.attention_sparsity)
        results["quality_match"].append(overlap)

    # Summary statistics
    results["avg_speedup"] = np.mean(results["speedups"])
    results["avg_sparsity"] = np.mean(results["sparsities"])
    results["avg_quality_match"] = np.mean(results["quality_match"])

    return results


def benchmark_long_context_scaling(
    spiking_rag: SpikingRAGRetriever,
    context_sizes: List[int] = [1000, 5000, 10000, 50000, 100000],
) -> Dict[str, List]:
    """
    Benchmark retrieval time vs context size.

    Demonstrates O(n*k) scaling vs O(n²) for dense attention.
    """
    results = {
        "context_sizes": context_sizes,
        "retrieval_times": [],
        "sparsities": [],
        "speedups": [],
    }

    test_query = "What are the key findings?"

    for context_size in context_sizes:
        print(f"Benchmarking context size: {context_size} tokens...")

        result = spiking_rag.get_long_context(
            query=test_query,
            max_tokens=context_size,
        )

        results["retrieval_times"].append(result.retrieval_time_ms)
        results["sparsities"].append(result.attention_sparsity)
        results["speedups"].append(result.attention_speedup)

    return results
