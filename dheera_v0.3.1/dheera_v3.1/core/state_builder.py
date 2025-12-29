# core/state_builder.py
"""
Dheera v0.3.0 - Enhanced State Builder
Builds 64-dimensional state vector for Rainbow DQN.

State dimensions:
[0-15]   Original features (sentiment, complexity, etc.)
[16-31]  Semantic embedding features (compressed from 384-dim)
[32-47]  Cognitive features (intent, entities, dialogue state)
[48-63]  Context features (RAG, memory, history)
"""

import numpy as np
from typing import Optional, Dict, Any, List
import re


class StateBuilder:
    """
    Builds enhanced 64-dimensional state vector.
    
    Combines:
    - Traditional hand-crafted features (16)
    - Semantic embedding features (16)
    - Cognitive layer features (16)
    - Context/memory features (16)
    """
    
    STATE_DIM = 64
    
    # Feature indices
    IDX_ORIGINAL = slice(0, 16)
    IDX_SEMANTIC = slice(16, 32)
    IDX_COGNITIVE = slice(32, 48)
    IDX_CONTEXT = slice(48, 64)
    
    # Search trigger keywords
    SEARCH_TRIGGERS = [
        "search", "find", "look up", "google", "latest", "current",
        "today", "news", "recent", "update", "price", "weather",
        "who is", "what is the current", "how much is",
    ]
    
    # Code keywords
    CODE_KEYWORDS = [
        "code", "program", "script", "function", "class", "python",
        "javascript", "bug", "error", "debug", "compile", "syntax",
    ]
    
    # Math keywords
    MATH_KEYWORDS = [
        "calculate", "compute", "sum", "add", "subtract", "multiply",
        "divide", "equation", "formula", "math", "percentage",
    ]
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        intent_classifier: Optional[Any] = None,
        entity_extractor: Optional[Any] = None,
        dialogue_tracker: Optional[Any] = None,
        working_memory: Optional[Any] = None,
    ):
        self.embedding_model = embedding_model
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.dialogue_tracker = dialogue_tracker
        self.working_memory = working_memory
        
        # Embedding projection matrix (384 -> 16)
        self._projection = None
        if embedding_model:
            emb_dim = getattr(embedding_model, 'embedding_dim', 384)
            self._projection = np.random.randn(emb_dim, 16).astype(np.float32)
            self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)
    
    def build_state(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None,
        rag_results: Optional[List] = None,
    ) -> np.ndarray:
        """
        Build 64-dimensional state vector.
        
        Args:
            user_message: Current user message
            context: Additional context
            conversation_history: Recent conversation turns
            rag_results: Retrieved RAG documents
            
        Returns:
            State vector (64,)
        """
        context = context or {}
        conversation_history = conversation_history or []
        rag_results = rag_results or []
        
        # Initialize state
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        
        # 1. Original features [0-15]
        state[self.IDX_ORIGINAL] = self._build_original_features(
            user_message, context, conversation_history
        )
        
        # 2. Semantic features [16-31]
        state[self.IDX_SEMANTIC] = self._build_semantic_features(user_message)
        
        # 3. Cognitive features [32-47]
        state[self.IDX_COGNITIVE] = self._build_cognitive_features(
            user_message, context
        )
        
        # 4. Context features [48-63]
        state[self.IDX_CONTEXT] = self._build_context_features(
            conversation_history, rag_results, context
        )
        
        return state
    
    def _build_original_features(
        self,
        message: str,
        context: Dict,
        history: List[Dict],
    ) -> np.ndarray:
        """Build original 16 hand-crafted features."""
        features = np.zeros(16, dtype=np.float32)
        
        message_lower = message.lower()
        words = message.split()
        
        # [0] Conversation length (normalized)
        features[0] = min(len(history) / 20.0, 1.0)
        
        # [1] Sentiment (-1 to 1)
        features[1] = self._estimate_sentiment(message_lower)
        
        # [2] Complexity (0 to 1)
        features[2] = self._estimate_complexity(message)
        
        # [3] Uncertainty (0 to 1)
        features[3] = self._estimate_uncertainty(message_lower)
        
        # [4] Frustration (0 to 1)
        features[4] = self._estimate_frustration(message_lower)
        
        # [5] Requires code (0 or 1)
        features[5] = float(any(kw in message_lower for kw in self.CODE_KEYWORDS))
        
        # [6] Requires math (0 or 1)
        features[6] = float(any(kw in message_lower for kw in self.MATH_KEYWORDS))
        
        # [7] Tools available (from context)
        features[7] = float(context.get("tools_available", True))
        
        # [8] Search intent (0 to 1)
        features[8] = self._estimate_search_intent(message_lower)
        
        # [9] Is simple query (0 or 1)
        features[9] = float(len(words) < 5 and not message.endswith("?"))
        
        # [10] Question score (0 to 1)
        features[10] = self._estimate_question_score(message)
        
        # [11] Recency need (0 to 1)
        features[11] = self._estimate_recency_need(message_lower)
        
        # [12-15] Topic embedding (simple)
        topic_emb = self._simple_topic_embedding(message_lower)
        features[12:16] = topic_emb
        
        return features
    
    def _build_semantic_features(self, message: str) -> np.ndarray:
        """Build semantic embedding features (16-dim)."""
        if self.embedding_model is None or self._projection is None:
            # Fallback: hash-based features
            return self._fallback_semantic_features(message)
        
        # Get full embedding
        embedding = self.embedding_model.embed(message)
        
        # Project to 16 dimensions
        projected = np.dot(embedding, self._projection)
        
        # Normalize
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        
        return projected.astype(np.float32)
    
    def _build_cognitive_features(
        self,
        message: str,
        context: Dict,
    ) -> np.ndarray:
        """Build cognitive layer features (16-dim)."""
        features = np.zeros(16, dtype=np.float32)
        
        # Intent features [0-5]
        if self.intent_classifier:
            intent_result = self.intent_classifier.classify(message)
            features[0] = intent_result.confidence
            
            # One-hot for top intents
            intent_map = {
                "greeting": 1, "question_factual": 2, "request_action": 3,
                "request_search": 4, "feedback_positive": 5, "feedback_negative": 6,
            }
            intent_idx = intent_map.get(intent_result.primary_intent.value, 0)
            if 1 <= intent_idx <= 5:
                features[intent_idx] = 1.0
        else:
            # Simple intent estimation
            features[0] = 0.5
            if message.endswith("?"):
                features[2] = 1.0  # Question
        
        # Entity features [6-10]
        if self.entity_extractor:
            entity_result = self.entity_extractor.extract(message)
            summary = self.entity_extractor.get_entity_summary(entity_result)
            
            features[6] = min(summary.get("entity_count", 0) / 5.0, 1.0)
            features[7] = float(summary.get("has_code_entities", False))
            features[8] = float(summary.get("has_numbers", False))
            features[9] = float(summary.get("has_datetime", False))
            features[10] = min(len(summary.get("action_verbs", [])) / 3.0, 1.0)
        else:
            # Simple entity estimation
            features[6] = min(len(re.findall(r'\b[A-Z][a-z]+\b', message)) / 5.0, 1.0)
            features[8] = float(bool(re.search(r'\d', message)))
        
        # Dialogue state features [11-15]
        if self.dialogue_tracker:
            state_features = self.dialogue_tracker.get_state_features()
            features[11] = state_features.get("user_satisfaction", 0.5)
            features[12] = state_features.get("user_engagement", 0.5)
            features[13] = float(state_features.get("task_in_progress", False))
            features[14] = float(state_features.get("needs_clarification", False))
            features[15] = state_features.get("turn_count_normalized", 0.0)
        else:
            features[11] = 0.5  # Default satisfaction
            features[12] = 0.5  # Default engagement
        
        return features
    
    def _build_context_features(
        self,
        history: List[Dict],
        rag_results: List,
        context: Dict,
    ) -> np.ndarray:
        """Build context/memory features (16-dim)."""
        features = np.zeros(16, dtype=np.float32)
        
        # History features [0-5]
        features[0] = min(len(history) / 10.0, 1.0)  # History length
        
        if history:
            # Recent reward trend
            recent_rewards = [h.get("reward", 0) for h in history[-5:]]
            features[1] = np.mean(recent_rewards) if recent_rewards else 0.0
            
            # Recent action diversity
            recent_actions = [h.get("action", 0) for h in history[-5:]]
            features[2] = len(set(recent_actions)) / 8.0 if recent_actions else 0.0
            
            # Average message length trend
            recent_lengths = [len(h.get("user", "").split()) for h in history[-3:]]
            features[3] = np.mean(recent_lengths) / 20.0 if recent_lengths else 0.0
        
        # RAG features [6-10]
        features[6] = min(len(rag_results) / 5.0, 1.0)  # RAG result count
        
        if rag_results:
            # Average RAG score
            scores = [r.score if hasattr(r, 'score') else 0.5 for r in rag_results]
            features[7] = np.mean(scores) if scores else 0.0
            
            # RAG diversity (different sources)
            sources = set(
                r.metadata.get('source', 'unknown') if hasattr(r, 'metadata') else 'unknown'
                for r in rag_results
            )
            features[8] = len(sources) / 3.0
        
        # Working memory features [11-15]
        if self.working_memory:
            mem_features = self.working_memory.get_state_features()
            features[11] = mem_features.get("has_active_task", 0.0)
            features[12] = mem_features.get("task_progress", 0.0)
            features[13] = mem_features.get("has_retrieved_context", 0.0)
            features[14] = mem_features.get("has_tool_results", 0.0)
            features[15] = mem_features.get("memory_item_count", 0.0)
        
        # Context flags
        features[9] = float(context.get("search_performed", False))
        features[10] = float(context.get("tool_used", False))
        
        return features
    
    def _fallback_semantic_features(self, message: str) -> np.ndarray:
        """Fallback semantic features without embedding model."""
        import hashlib
        
        features = np.zeros(16, dtype=np.float32)
        message_lower = message.lower()
        words = message_lower.split()
        
        # Hash-based features
        for i, word in enumerate(words[:8]):
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = i % 16
            features[idx] += (word_hash % 100) / 100.0
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _estimate_sentiment(self, message: str) -> float:
        """Estimate message sentiment."""
        positive = ["thanks", "great", "good", "awesome", "helpful", "perfect", "love"]
        negative = ["bad", "wrong", "hate", "terrible", "awful", "useless", "frustrated"]
        
        pos_count = sum(1 for w in positive if w in message)
        neg_count = sum(1 for w in negative if w in message)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _estimate_complexity(self, message: str) -> float:
        """Estimate message complexity."""
        words = message.split()
        
        # Factors: length, avg word length, punctuation
        length_score = min(len(words) / 30.0, 1.0)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        word_len_score = min(avg_word_len / 10.0, 1.0)
        punct_score = len(re.findall(r'[,;:\(\)\[\]]', message)) / 10.0
        
        return (length_score + word_len_score + punct_score) / 3.0
    
    def _estimate_uncertainty(self, message: str) -> float:
        """Estimate uncertainty in message."""
        uncertainty_words = [
            "maybe", "perhaps", "might", "could", "possibly", "unsure",
            "not sure", "don't know", "confused", "unclear",
        ]
        
        count = sum(1 for w in uncertainty_words if w in message)
        return min(count / 3.0, 1.0)
    
    def _estimate_frustration(self, message: str) -> float:
        """Estimate user frustration."""
        frustration_signals = [
            "again", "still", "already told", "not working", "doesn't work",
            "wrong", "!!!",  "why", "frustrated", "annoying",
        ]
        
        count = sum(1 for s in frustration_signals if s in message)
        return min(count / 3.0, 1.0)
    
    def _estimate_search_intent(self, message: str) -> float:
        """Estimate intent to search."""
        count = sum(1 for trigger in self.SEARCH_TRIGGERS if trigger in message)
        return min(count / 2.0, 1.0)
    
    def _estimate_question_score(self, message: str) -> float:
        """Estimate how much this is a question."""
        score = 0.0
        
        if message.endswith("?"):
            score += 0.5
        
        question_starters = ["what", "who", "where", "when", "why", "how", "is", "are", "can", "could", "would"]
        if any(message.lower().startswith(q) for q in question_starters):
            score += 0.5
        
        return min(score, 1.0)
    
    def _estimate_recency_need(self, message: str) -> float:
        """Estimate need for recent information."""
        recency_words = [
            "today", "now", "current", "latest", "recent", "new",
            "2024", "2025", "this week", "this month",
        ]
        
        count = sum(1 for w in recency_words if w in message)
        return min(count / 2.0, 1.0)
    
    def _simple_topic_embedding(self, message: str) -> np.ndarray:
        """Simple 4-dim topic embedding."""
        topics = np.zeros(4, dtype=np.float32)
        
        # [0] Code/tech
        topics[0] = float(any(w in message for w in ["code", "python", "javascript", "program", "bug", "api"]))
        
        # [1] Math/numbers
        topics[1] = float(any(w in message for w in ["calculate", "math", "number", "sum"]) or bool(re.search(r'\d+', message)))
        
        # [2] Factual/info
        topics[2] = float(any(w in message for w in ["what is", "who is", "where", "when", "fact"]))
        
        # [3] Creative/open
        topics[3] = float(any(w in message for w in ["write", "create", "story", "poem", "imagine"]))
        
        return topics
    
    def should_search(self, state: np.ndarray) -> bool:
        """Determine if search should be performed based on state."""
        search_intent = state[8]  # Search intent feature
        recency_need = state[11]  # Recency need feature
        
        return search_intent > 0.5 or recency_need > 0.5


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing StateBuilder...")
    
    builder = StateBuilder()
    
    test_messages = [
        "Hello, how are you?",
        "What is the latest news about AI?",
        "Calculate 25 * 4 + 10",
        "Help me write a Python script to read CSV files",
        "Why is the sky blue?",
        "I'm frustrated, this still doesn't work!!!",
    ]
    
    print("\nState Building Tests:")
    print("-" * 60)
    
    for msg in test_messages:
        state = builder.build_state(
            user_message=msg,
            conversation_history=[{"user": "Hi", "assistant": "Hello!"}],
        )
        
        print(f"\n'{msg}'")
        print(f"  State shape: {state.shape}")
        print(f"  Original [0-3]: {state[0:4]}")
        print(f"  Search intent: {state[8]:.2f}")
        print(f"  Should search: {builder.should_search(state)}")
        print(f"  Code needed: {state[5]:.0f}, Math needed: {state[6]:.0f}")
    
    # Verify dimensions
    assert builder.STATE_DIM == 64
    assert state.shape == (64,)
    print(f"\n✓ State dimension verified: {builder.STATE_DIM}")
    
    print("\n✅ State builder tests passed!")
