"""
Dheera State Builder
Converts chat context into a fixed-size state vector for DQN.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ConversationContext:
    """Holds the current conversation state."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    user_feedback_history: List[float] = field(default_factory=list)
    tools_available: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateBuilder:
    """
    Builds a fixed-size state vector from conversation context.
    This vector is fed to the DQN for action selection.
    
    State dimensions (default 16):
    - [0]     : Normalized conversation length
    - [1]     : User sentiment (-1 to 1)
    - [2]     : Query complexity (0 to 1)
    - [3]     : Uncertainty score (0 to 1)
    - [4]     : Frustration level (0 to 1)
    - [5]     : Requires code (0 or 1)
    - [6]     : Requires math (0 or 1)
    - [7]     : Tools available (0 or 1)
    - [8-15]  : Topic embedding (8 dims)
    """
    
    # Topic keywords for simple classification
    TOPIC_KEYWORDS = {
        "code": ["code", "python", "script", "function", "class", "programming", "bug", "error"],
        "math": ["math", "calculate", "equation", "formula", "number", "statistics"],
        "explain": ["explain", "what is", "how does", "why", "understand", "meaning"],
        "plan": ["plan", "steps", "roadmap", "strategy", "approach", "design"],
        "debug": ["error", "bug", "fix", "issue", "problem", "broken", "wrong"],
        "create": ["create", "build", "make", "generate", "write", "develop"],
        "compare": ["compare", "difference", "versus", "vs", "better", "choose"],
        "summarize": ["summarize", "summary", "brief", "short", "tldr", "quick"],
    }
    
    # Sentiment indicators
    POSITIVE_WORDS = ["good", "great", "thanks", "helpful", "perfect", "excellent", "awesome", "love"]
    NEGATIVE_WORDS = ["bad", "wrong", "no", "not", "confused", "frustrat", "annoying", "hate", "useless"]
    UNCERTAINTY_WORDS = ["maybe", "perhaps", "not sure", "confused", "unclear", "don't know", "?"]
    
    def __init__(self, state_dim: int = 16):
        self.state_dim = state_dim
        self.topic_dim = 8  # Fixed topic embedding size
        
    def build_state(self, context: ConversationContext) -> np.ndarray:
        """
        Convert conversation context to state vector.
        
        Args:
            context: ConversationContext with messages and metadata
            
        Returns:
            np.ndarray of shape (state_dim,)
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Get latest user message
        last_user_msg = self._get_last_user_message(context)
        all_user_text = self._get_all_user_text(context)
        
        # [0] Conversation length (normalized, max 20 turns)
        state[0] = min(context.turn_count / 20.0, 1.0)
        
        # [1] User sentiment
        state[1] = self._compute_sentiment(last_user_msg)
        
        # [2] Query complexity
        state[2] = self._compute_complexity(last_user_msg)
        
        # [3] Uncertainty score
        state[3] = self._compute_uncertainty(last_user_msg)
        
        # [4] Frustration level
        state[4] = self._compute_frustration(context)
        
        # [5] Requires code
        state[5] = self._requires_code(last_user_msg)
        
        # [6] Requires math
        state[6] = self._requires_math(last_user_msg)
        
        # [7] Tools available
        state[7] = 1.0 if len(context.tools_available) > 0 else 0.0
        
        # [8-15] Topic embedding
        topic_vec = self._compute_topic_embedding(last_user_msg)
        state[8:16] = topic_vec
        
        return state
    
    def _get_last_user_message(self, context: ConversationContext) -> str:
        """Extract the most recent user message."""
        for msg in reversed(context.messages):
            if msg.get("role") == "user":
                return msg.get("content", "").lower()
        return ""
    
    def _get_all_user_text(self, context: ConversationContext) -> str:
        """Concatenate all user messages."""
        texts = []
        for msg in context.messages:
            if msg.get("role") == "user":
                texts.append(msg.get("content", ""))
        return " ".join(texts).lower()
    
    def _compute_sentiment(self, text: str) -> float:
        """
        Simple sentiment score from -1 (negative) to 1 (positive).
        """
        pos_count = sum(1 for word in self.POSITIVE_WORDS if word in text)
        neg_count = sum(1 for word in self.NEGATIVE_WORDS if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _compute_complexity(self, text: str) -> float:
        """
        Estimate query complexity based on length and structure.
        """
        if not text:
            return 0.0
        
        # Factors: length, question marks, technical terms
        word_count = len(text.split())
        question_marks = text.count("?")
        
        # Normalize (assume max ~100 words is complex)
        length_score = min(word_count / 100.0, 1.0)
        
        # Multiple questions increase complexity
        question_score = min(question_marks / 3.0, 1.0)
        
        # Check for technical indicators
        technical_words = ["implement", "algorithm", "architecture", "optimize", "debug", "deploy"]
        tech_score = min(sum(1 for w in technical_words if w in text) / 3.0, 1.0)
        
        return (length_score * 0.4 + question_score * 0.3 + tech_score * 0.3)
    
    def _compute_uncertainty(self, text: str) -> float:
        """Detect user uncertainty/confusion."""
        uncertainty_count = sum(1 for phrase in self.UNCERTAINTY_WORDS if phrase in text)
        return min(uncertainty_count / 3.0, 1.0)
    
    def _compute_frustration(self, context: ConversationContext) -> float:
        """
        Estimate frustration from feedback history and conversation patterns.
        """
        if not context.user_feedback_history:
            return 0.0
        
        # Recent negative feedback increases frustration
        recent = context.user_feedback_history[-5:]
        negative_ratio = sum(1 for f in recent if f < 0) / len(recent)
        
        # Repeated questions also indicate frustration
        # (simplified: just use negative ratio for now)
        return negative_ratio
    
    def _requires_code(self, text: str) -> float:
        """Check if query likely needs code."""
        code_indicators = ["code", "script", "function", "implement", "python", "javascript", 
                          "write a", "create a", "build a", "```"]
        return 1.0 if any(ind in text for ind in code_indicators) else 0.0
    
    def _requires_math(self, text: str) -> float:
        """Check if query involves math/calculations."""
        math_indicators = ["calculate", "math", "equation", "formula", "compute", 
                          "average", "sum", "percentage", "probability"]
        return 1.0 if any(ind in text for ind in math_indicators) else 0.0
    
    def _compute_topic_embedding(self, text: str) -> np.ndarray:
        """
        Simple topic embedding based on keyword matching.
        Returns 8-dim vector representing topic distribution.
        """
        embedding = np.zeros(8, dtype=np.float32)
        
        topic_names = list(self.TOPIC_KEYWORDS.keys())
        
        for i, topic in enumerate(topic_names):
            keywords = self.TOPIC_KEYWORDS[topic]
            match_count = sum(1 for kw in keywords if kw in text)
            embedding[i] = min(match_count / 3.0, 1.0)
        
        # Normalize if non-zero
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def state_to_dict(self, state: np.ndarray) -> Dict[str, float]:
        """Convert state vector to readable dictionary."""
        return {
            "conversation_length": state[0],
            "sentiment": state[1],
            "complexity": state[2],
            "uncertainty": state[3],
            "frustration": state[4],
            "requires_code": state[5],
            "requires_math": state[6],
            "tools_available": state[7],
            "topic_code": state[8],
            "topic_math": state[9],
            "topic_explain": state[10],
            "topic_plan": state[11],
            "topic_debug": state[12],
            "topic_create": state[13],
            "topic_compare": state[14],
            "topic_summarize": state[15],
        }


# Quick test
if __name__ == "__main__":
    builder = StateBuilder()
    
    # Simulate a conversation
    context = ConversationContext(
        messages=[
            {"role": "user", "content": "Can you help me write a Python script to calculate fibonacci numbers?"},
        ],
        turn_count=1,
        tools_available=["python_executor"],
    )
    
    state = builder.build_state(context)
    print(f"State vector: {state}")
    print(f"\nState breakdown:")
    for key, value in builder.state_to_dict(state).items():
        print(f"  {key}: {value:.3f}")
