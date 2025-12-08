# core/state_builder.py
"""
Dheera State Builder
Converts chat context into a fixed-size state vector for DQN.
Version 0.2.0 - Enhanced with search detection and better feature extraction.
"""

import re
import math
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationContext:
    """Holds the current conversation state."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    user_feedback_history: List[float] = field(default_factory=list)
    tools_available: List[str] = field(default_factory=list)
    search_available: bool = True
    last_action: Optional[int] = None
    last_search_results: Optional[Dict] = None
    session_start: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
        if role == "user":
            self.turn_count += 1
    
    def get_last_user_message(self) -> str:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    
    def get_last_assistant_message(self) -> str:
        """Get the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    
    def clear(self):
        """Reset the context."""
        self.messages = []
        self.turn_count = 0
        self.user_feedback_history = []
        self.last_action = None
        self.last_search_results = None


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
    - [8]     : Search intent (0 to 1) - NEW
    - [9]     : Is greeting/simple (0 or 1) - NEW
    - [10]    : Question type score (0 to 1) - NEW
    - [11]    : Recency need (0 to 1) - NEW
    - [12-15] : Topic embedding (4 dims)
    """
    
    # Topic keywords for classification
    TOPIC_KEYWORDS = {
        "code": ["code", "python", "script", "function", "class", "programming", 
                 "bug", "error", "javascript", "java", "rust", "golang", "api"],
        "math": ["math", "calculate", "equation", "formula", "number", "statistics",
                 "sum", "average", "percentage", "probability", "compute"],
        "factual": ["what is", "who is", "where is", "when did", "how many",
                    "define", "meaning", "definition", "fact", "true"],
        "creative": ["write", "create", "generate", "story", "poem", "idea",
                     "brainstorm", "imagine", "design", "invent"],
    }
    
    # Search trigger keywords
    SEARCH_TRIGGERS = [
        "search", "find", "look up", "google", "what is the latest",
        "current", "today", "news", "recent", "2024", "2025",
        "who is", "where is", "price of", "weather", "stock",
        "score", "result", "update", "latest version", "new",
        "how much does", "what happened", "tell me about",
    ]
    
    # Recency indicators (things that change over time)
    RECENCY_INDICATORS = [
        "latest", "current", "now", "today", "recent", "new",
        "2024", "2025", "this year", "this month", "this week",
        "president", "ceo", "price", "weather", "score", "stock",
        "version", "update", "release", "news", "happening",
    ]
    
    # Greeting patterns
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|howdy|greetings|sup|yo)[\s!.]*$',
        r'^good\s*(morning|afternoon|evening|night)[\s!.]*$',
        r'^(what\'?s up|how are you|how\'?s it going)[\s?!.]*$',
    ]
    
    # Simple response patterns
    SIMPLE_PATTERNS = [
        r'^(yes|no|ok|okay|sure|thanks|thank you|cool|nice|great|good|bye|goodbye)[\s!.]*$',
        r'^(got it|understood|makes sense|i see|alright)[\s!.]*$',
    ]
    
    # Sentiment indicators
    POSITIVE_WORDS = [
        "good", "great", "thanks", "helpful", "perfect", "excellent", 
        "awesome", "love", "amazing", "wonderful", "fantastic", "best"
    ]
    NEGATIVE_WORDS = [
        "bad", "wrong", "no", "not", "confused", "frustrat", "annoying", 
        "hate", "useless", "terrible", "worst", "stupid", "broken"
    ]
    UNCERTAINTY_WORDS = [
        "maybe", "perhaps", "not sure", "confused", "unclear", 
        "don't know", "uncertain", "might", "possibly", "?"
    ]
    
    # Question word patterns
    QUESTION_WORDS = ["what", "who", "where", "when", "why", "how", "which", "can", "could", "would", "should"]
    
    def __init__(self, state_dim: int = 16):
        self.state_dim = state_dim
        self.topic_dim = 4  # Reduced topic embedding for more features
        
        # Compile regex patterns
        self._greeting_patterns = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        
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
        last_user_msg_lower = last_user_msg.lower()
        
        # [0] Conversation length (normalized, max 20 turns)
        state[0] = min(context.turn_count / 20.0, 1.0)
        
        # [1] User sentiment
        state[1] = self._compute_sentiment(last_user_msg_lower)
        
        # [2] Query complexity
        state[2] = self._compute_complexity(last_user_msg)
        
        # [3] Uncertainty score
        state[3] = self._compute_uncertainty(last_user_msg_lower)
        
        # [4] Frustration level
        state[4] = self._compute_frustration(context)
        
        # [5] Requires code
        state[5] = self._requires_code(last_user_msg_lower)
        
        # [6] Requires math
        state[6] = self._requires_math(last_user_msg_lower)
        
        # [7] Tools available
        state[7] = 1.0 if len(context.tools_available) > 0 or context.search_available else 0.0
        
        # [8] Search intent (NEW)
        state[8] = self._compute_search_intent(last_user_msg_lower)
        
        # [9] Is greeting/simple (NEW)
        state[9] = self._is_simple_message(last_user_msg)
        
        # [10] Question type score (NEW)
        state[10] = self._compute_question_score(last_user_msg_lower)
        
        # [11] Recency need (NEW)
        state[11] = self._compute_recency_need(last_user_msg_lower)
        
        # [12-15] Topic embedding (4 dims)
        topic_vec = self._compute_topic_embedding(last_user_msg_lower)
        state[12:16] = topic_vec
        
        return state
    
    def _get_last_user_message(self, context: ConversationContext) -> str:
        """Extract the most recent user message."""
        for msg in reversed(context.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
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
        if not text:
            return 0.0
            
        pos_count = sum(1 for word in self.POSITIVE_WORDS if word in text)
        neg_count = sum(1 for word in self.NEGATIVE_WORDS if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _compute_complexity(self, text: str) -> float:
        """
        Estimate query complexity based on length, structure, and content.
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Factors: length, question marks, technical terms, sentence count
        word_count = len(text.split())
        question_marks = text.count("?")
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Normalize (assume max ~100 words is complex)
        length_score = min(word_count / 50.0, 1.0)
        
        # Multiple questions increase complexity
        question_score = min(question_marks / 3.0, 1.0)
        
        # Multiple sentences increase complexity
        sentence_score = min(sentence_count / 5.0, 1.0)
        
        # Check for technical indicators
        technical_words = [
            "implement", "algorithm", "architecture", "optimize", "debug", 
            "deploy", "configure", "integrate", "refactor", "analyze"
        ]
        tech_score = min(sum(1 for w in technical_words if w in text_lower) / 3.0, 1.0)
        
        # Combine scores
        complexity = (
            length_score * 0.3 + 
            question_score * 0.2 + 
            sentence_score * 0.2 + 
            tech_score * 0.3
        )
        
        return min(complexity, 1.0)
    
    def _compute_uncertainty(self, text: str) -> float:
        """Detect user uncertainty/confusion."""
        if not text:
            return 0.0
            
        uncertainty_count = sum(1 for phrase in self.UNCERTAINTY_WORDS if phrase in text)
        
        # Question marks also indicate uncertainty
        question_marks = text.count("?")
        
        score = (uncertainty_count / 3.0) + (question_marks / 5.0) * 0.5
        return min(score, 1.0)
    
    def _compute_frustration(self, context: ConversationContext) -> float:
        """
        Estimate frustration from feedback history and conversation patterns.
        """
        frustration = 0.0
        
        # Recent negative feedback increases frustration
        if context.user_feedback_history:
            recent = context.user_feedback_history[-5:]
            negative_ratio = sum(1 for f in recent if f < 0) / len(recent)
            frustration += negative_ratio * 0.5
        
        # Long conversations might indicate frustration
        if context.turn_count > 10:
            frustration += min((context.turn_count - 10) / 20.0, 0.3)
        
        # Check for frustration words in recent messages
        last_msg = self._get_last_user_message(context).lower()
        frustration_words = ["frustrated", "annoying", "not working", "again", "still", "why isn't"]
        frustration_count = sum(1 for w in frustration_words if w in last_msg)
        frustration += min(frustration_count / 3.0, 0.3)
        
        return min(frustration, 1.0)
    
    def _requires_code(self, text: str) -> float:
        """Check if query likely needs code."""
        code_indicators = [
            "code", "script", "function", "implement", "python", "javascript",
            "write a", "create a", "build a", "```", "program", "class",
            "method", "variable", "loop", "array", "list", "dictionary"
        ]
        
        match_count = sum(1 for ind in code_indicators if ind in text)
        return min(match_count / 2.0, 1.0)
    
    def _requires_math(self, text: str) -> float:
        """Check if query involves math/calculations."""
        math_indicators = [
            "calculate", "math", "equation", "formula", "compute",
            "average", "sum", "percentage", "probability", "multiply",
            "divide", "add", "subtract", "total", "mean", "median"
        ]
        
        # Also check for numbers and operators
        has_numbers = bool(re.search(r'\d+', text))
        has_operators = bool(re.search(r'[+\-*/=]', text))
        
        match_count = sum(1 for ind in math_indicators if ind in text)
        score = match_count / 2.0
        
        if has_numbers:
            score += 0.2
        if has_operators:
            score += 0.2
            
        return min(score, 1.0)
    
    def _compute_search_intent(self, text: str) -> float:
        """
        Detect intent to search for information.
        High score = user likely wants current/external information.
        """
        if not text:
            return 0.0
        
        # Check for explicit search triggers
        trigger_count = sum(1 for trigger in self.SEARCH_TRIGGERS if trigger in text)
        trigger_score = min(trigger_count / 2.0, 1.0)
        
        # Check for factual question patterns
        factual_patterns = [
            r'\bwhat is\b', r'\bwho is\b', r'\bwhere is\b', r'\bwhen did\b',
            r'\bhow much\b', r'\bhow many\b', r'\bwhat are\b', r'\btell me about\b'
        ]
        factual_score = sum(0.3 for p in factual_patterns if re.search(p, text))
        factual_score = min(factual_score, 1.0)
        
        # Combine scores
        search_intent = max(trigger_score, factual_score)
        
        # Reduce if it's clearly a coding/creative task
        if self._requires_code(text) > 0.7:
            search_intent *= 0.5
        
        return min(search_intent, 1.0)
    
    def _is_simple_message(self, text: str) -> float:
        """
        Check if message is a simple greeting or acknowledgment.
        """
        if not text:
            return 0.0
        
        text_stripped = text.strip()
        
        # Check greeting patterns
        for pattern in self._greeting_patterns:
            if pattern.match(text_stripped):
                return 1.0
        
        # Check simple response patterns
        for pattern in self._simple_patterns:
            if pattern.match(text_stripped):
                return 1.0
        
        # Very short messages are often simple
        word_count = len(text_stripped.split())
        if word_count <= 2:
            return 0.8
        elif word_count <= 4:
            return 0.4
        
        return 0.0
    
    def _compute_question_score(self, text: str) -> float:
        """
        Compute question type score.
        Higher = more likely a direct question needing an answer.
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # Check for question words at start
        first_word = text.split()[0] if text.split() else ""
        if first_word in self.QUESTION_WORDS:
            score += 0.5
        
        # Question marks
        if "?" in text:
            score += 0.3
        
        # "Can you" / "Could you" patterns
        if re.search(r'\b(can|could|would|will) you\b', text):
            score += 0.3
        
        # "How to" patterns
        if re.search(r'\bhow (to|do|can|should)\b', text):
            score += 0.4
        
        return min(score, 1.0)
    
    def _compute_recency_need(self, text: str) -> float:
        """
        Compute how much the query needs recent/current information.
        High score = needs up-to-date info (good candidate for search).
        """
        if not text:
            return 0.0
        
        # Check for recency indicators
        recency_count = sum(1 for ind in self.RECENCY_INDICATORS if ind in text)
        recency_score = min(recency_count / 2.0, 1.0)
        
        # Check for year mentions (recent years = high recency)
        year_match = re.search(r'\b(202[3-9]|2030)\b', text)
        if year_match:
            recency_score = max(recency_score, 0.8)
        
        # Check for "current" type questions
        current_patterns = [
            r'\bcurrent(ly)?\b', r'\bright now\b', r'\btoday\b',
            r'\bthis (year|month|week)\b', r'\blatest\b'
        ]
        for pattern in current_patterns:
            if re.search(pattern, text):
                recency_score = max(recency_score, 0.7)
                break
        
        return min(recency_score, 1.0)
    
    def _compute_topic_embedding(self, text: str) -> np.ndarray:
        """
        Simple topic embedding based on keyword matching.
        Returns 4-dim vector representing topic distribution.
        """
        embedding = np.zeros(self.topic_dim, dtype=np.float32)
        
        if not text:
            return embedding
        
        topic_names = list(self.TOPIC_KEYWORDS.keys())
        
        for i, topic in enumerate(topic_names[:self.topic_dim]):
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
            "conversation_length": float(state[0]),
            "sentiment": float(state[1]),
            "complexity": float(state[2]),
            "uncertainty": float(state[3]),
            "frustration": float(state[4]),
            "requires_code": float(state[5]),
            "requires_math": float(state[6]),
            "tools_available": float(state[7]),
            "search_intent": float(state[8]),
            "is_simple": float(state[9]),
            "question_score": float(state[10]),
            "recency_need": float(state[11]),
            "topic_code": float(state[12]),
            "topic_math": float(state[13]),
            "topic_factual": float(state[14]),
            "topic_creative": float(state[15]),
        }
    
    def get_action_suggestions(self, state: np.ndarray) -> Dict[int, float]:
        """
        Get action suggestions based on state features.
        Used to guide DQN exploration early in training.
        
        Returns:
            Dict mapping action_id to suggestion score (0-1)
        """
        # Action IDs (must match action_space.py)
        DIRECT_RESPONSE = 0
        CLARIFY_QUESTION = 1
        USE_TOOL = 2
        SEARCH_WEB = 3
        BREAK_DOWN_TASK = 4
        REFLECT_AND_REASON = 5
        DEFER_OR_DECLINE = 6
        
        suggestions = {i: 0.0 for i in range(7)}
        
        # Extract state features
        is_simple = state[9]
        search_intent = state[8]
        recency_need = state[11]
        complexity = state[2]
        uncertainty = state[3]
        requires_code = state[5]
        requires_math = state[6]
        question_score = state[10]
        
        # Simple messages -> DIRECT_RESPONSE
        if is_simple > 0.7:
            suggestions[DIRECT_RESPONSE] = 0.9
            return suggestions
        
        # High search intent or recency need -> SEARCH_WEB
        if search_intent > 0.5 or recency_need > 0.5:
            suggestions[SEARCH_WEB] = max(search_intent, recency_need)
            suggestions[DIRECT_RESPONSE] = 0.2
        
        # High complexity -> BREAK_DOWN_TASK or REFLECT
        if complexity > 0.6:
            suggestions[BREAK_DOWN_TASK] = complexity * 0.7
            suggestions[REFLECT_AND_REASON] = complexity * 0.5
        
        # High uncertainty -> CLARIFY_QUESTION
        if uncertainty > 0.5:
            suggestions[CLARIFY_QUESTION] = uncertainty * 0.8
        
        # Requires code -> USE_TOOL or BREAK_DOWN
        if requires_code > 0.5:
            suggestions[USE_TOOL] = requires_code * 0.5
            suggestions[BREAK_DOWN_TASK] = requires_code * 0.4
        
        # Requires math -> USE_TOOL
        if requires_math > 0.5:
            suggestions[USE_TOOL] = max(suggestions[USE_TOOL], requires_math * 0.7)
        
        # Direct questions -> DIRECT_RESPONSE
        if question_score > 0.5 and not (search_intent > 0.5 or recency_need > 0.5):
            suggestions[DIRECT_RESPONSE] = max(suggestions[DIRECT_RESPONSE], question_score * 0.6)
        
        # Normalize to ensure at least one suggestion
        if max(suggestions.values()) == 0:
            suggestions[DIRECT_RESPONSE] = 0.5
        
        return suggestions
    
    def should_search(self, context: ConversationContext) -> bool:
        """
        Quick check if current context suggests web search.
        """
        state = self.build_state(context)
        return state[8] > 0.5 or state[11] > 0.5  # search_intent or recency_need
    
    def get_state_summary(self, state: np.ndarray) -> str:
        """Get a human-readable summary of the state."""
        state_dict = self.state_to_dict(state)
        
        summary_parts = []
        
        if state_dict["is_simple"] > 0.7:
            summary_parts.append("simple greeting/response")
        if state_dict["search_intent"] > 0.5:
            summary_parts.append("search intent")
        if state_dict["recency_need"] > 0.5:
            summary_parts.append("needs current info")
        if state_dict["complexity"] > 0.6:
            summary_parts.append("complex query")
        if state_dict["requires_code"] > 0.5:
            summary_parts.append("code-related")
        if state_dict["requires_math"] > 0.5:
            summary_parts.append("math-related")
        if state_dict["uncertainty"] > 0.5:
            summary_parts.append("user uncertain")
        if state_dict["frustration"] > 0.3:
            summary_parts.append("user frustrated")
        
        if not summary_parts:
            summary_parts.append("standard query")
        
        return ", ".join(summary_parts)


# ===============================
# Quick Test
# ===============================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing State Builder")
    print("=" * 60)
    
    builder = StateBuilder()
    
    # Test cases
    test_cases = [
        {
            "name": "Simple greeting",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        {
            "name": "Search query",
            "messages": [{"role": "user", "content": "What is the latest version of Python?"}],
        },
        {
            "name": "News search",
            "messages": [{"role": "user", "content": "Search for AI news today"}],
        },
        {
            "name": "Code request",
            "messages": [{"role": "user", "content": "Can you write a Python function to calculate fibonacci numbers?"}],
        },
        {
            "name": "Complex question",
            "messages": [{"role": "user", "content": "How do I implement a distributed system with microservices architecture? What are the best practices for handling failures and ensuring data consistency?"}],
        },
        {
            "name": "Math question",
            "messages": [{"role": "user", "content": "Calculate 25 * 4 + 100 / 5"}],
        },
        {
            "name": "Uncertain user",
            "messages": [{"role": "user", "content": "I'm not sure but maybe something about Python?"}],
        },
        {
            "name": "Factual question",
            "messages": [{"role": "user", "content": "Who is the CEO of Tesla?"}],
        },
        {
            "name": "Acknowledgment",
            "messages": [{"role": "user", "content": "Thanks!"}],
        },
    ]
    
    for test in test_cases:
        context = ConversationContext(
            messages=test["messages"],
            turn_count=1,
            tools_available=["calculator"],
            search_available=True,
        )
        
        state = builder.build_state(context)
        state_dict = builder.state_to_dict(state)
        suggestions = builder.get_action_suggestions(state)
        summary = builder.get_state_summary(state)
        
        print(f"\n📝 {test['name']}")
        print(f"   Message: \"{test['messages'][0]['content'][:50]}...\"" if len(test['messages'][0]['content']) > 50 else f"   Message: \"{test['messages'][0]['content']}\"")
        print(f"   Summary: {summary}")
        
        # Show key features
        key_features = {
            "search_intent": state_dict["search_intent"],
            "is_simple": state_dict["is_simple"],
            "complexity": state_dict["complexity"],
            "recency_need": state_dict["recency_need"],
        }
        print(f"   Features: {', '.join(f'{k}={v:.2f}' for k, v in key_features.items() if v > 0.1)}")
        
        # Show top action suggestions
        top_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[:3]
        action_names = ["DIRECT", "CLARIFY", "TOOL", "SEARCH", "BREAKDOWN", "REFLECT", "DEFER"]
        suggestion_strs = [f"{action_names[aid]}:{score:.2f}" for aid, score in top_suggestions if score > 0]
        if suggestion_strs:
            print(f"   Suggested: {', '.join(suggestion_strs)}")
        
        # Should search?
        if builder.should_search(context):
            print(f"   🔍 Should search: YES")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
