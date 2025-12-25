# cognitive/intent_classifier.py
"""
Dheera v0.3.0 - Intent Classifier
Classifies user intent to guide action selection.
"""

import re
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    """User intent categories."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    GRATITUDE = "gratitude"
    
    QUESTION_FACTUAL = "question_factual"
    QUESTION_OPINION = "question_opinion"
    QUESTION_HOW_TO = "question_how_to"
    QUESTION_WHY = "question_why"
    
    REQUEST_ACTION = "request_action"
    REQUEST_SEARCH = "request_search"
    REQUEST_CREATE = "request_create"
    REQUEST_EXPLAIN = "request_explain"
    REQUEST_CALCULATE = "request_calculate"
    
    STATEMENT = "statement"
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"
    
    CLARIFICATION = "clarification"
    CORRECTION = "correction"
    FOLLOWUP = "followup"
    
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    primary_intent: Intent
    confidence: float
    secondary_intents: List[Tuple[Intent, float]]
    raw_scores: Dict[str, float]
    features: Dict[str, bool]


class IntentClassifier:
    """
    Rule-based + pattern matching intent classifier.
    Fast and interpretable for Dheera's needs.
    """
    
    # Intent patterns (regex)
    PATTERNS = {
        Intent.GREETING: [
            r'^(hi|hello|hey|howdy|greetings|yo|sup)[\s!.,]*$',
            r'^good\s*(morning|afternoon|evening|night)[\s!.,]*$',
            r'^(what\'?s up|how are you|how\'?s it going)[\s?!.,]*$',
        ],
        
        Intent.FAREWELL: [
            r'^(bye|goodbye|see you|later|take care|cya)[\s!.,]*$',
            r'^(good\s*night|have a (good|nice) (day|one))[\s!.,]*$',
        ],
        
        Intent.GRATITUDE: [
            r'^(thanks|thank you|thx|ty|appreciate it)[\s!.,]*$',
            r'(thanks|thank you) (so much|a lot|very much)',
        ],
        
        Intent.FEEDBACK_POSITIVE: [
            r'^(great|awesome|perfect|excellent|good job|nice|cool|amazing)[\s!.,]*$',
            r'^(that\'?s? (great|perfect|exactly|right|correct))[\s!.,]*$',
            r'^(yes|yep|yeah|correct|exactly)[\s!.,]*$',
        ],
        
        Intent.FEEDBACK_NEGATIVE: [
            r'^(no|nope|wrong|incorrect|bad|not right)[\s!.,]*$',
            r'^(that\'?s? (wrong|incorrect|not (right|what i (asked|wanted))))[\s!.,]*$',
        ],
        
        Intent.REQUEST_SEARCH: [
            r'(search|find|look up|google|lookup)\s+(for\s+)?',
            r'(what is the latest|current|today\'?s?|recent)\s+',
            r'(news about|updates on)',
        ],
        
        Intent.REQUEST_CALCULATE: [
            r'(calculate|compute|what is|how much is)\s+[\d\+\-\*\/\(\)]+',
            r'(add|subtract|multiply|divide)\s+\d+',
            r'\d+\s*[\+\-\*\/\%]\s*\d+',
        ],
        
        Intent.REQUEST_CREATE: [
            r'^(write|create|generate|make|build|compose)\s+',
            r'^(can you|could you|please)\s+(write|create|generate|make)',
        ],
        
        Intent.REQUEST_EXPLAIN: [
            r'^(explain|describe|tell me about|what does .* mean)',
            r'^(can you|could you|please)\s+explain',
        ],
        
        Intent.QUESTION_HOW_TO: [
            r'^how (do|can|should|would|could) (i|you|we|one)',
            r'^(what\'?s the (best|right|correct) way to)',
            r'^(steps to|guide for|tutorial on)',
        ],
        
        Intent.QUESTION_WHY: [
            r'^why (is|are|do|does|did|would|should)',
            r'^(what is the reason|what causes)',
        ],
        
        Intent.QUESTION_FACTUAL: [
            r'^(what|who|where|when|which) (is|are|was|were|did)',
            r'^(how (many|much|long|old|far))',
            r'^(is|are|was|were|do|does|did|can|could|will|would)\s+\w+',
        ],
        
        Intent.QUESTION_OPINION: [
            r'^(what do you think|in your opinion|do you believe)',
            r'^(should i|would you recommend)',
            r'^(what\'?s? (better|best|worse|worst))',
        ],
        
        Intent.CLARIFICATION: [
            r'^(what do you mean|i don\'?t understand|can you clarify)',
            r'^(huh|what|pardon|sorry\?)',
            r'\?{2,}',
        ],
        
        Intent.CORRECTION: [
            r'^(no,?\s*(i meant|actually|i said))',
            r'^(let me rephrase|what i meant was)',
            r'^(correction|actually)',
        ],
        
        Intent.FOLLOWUP: [
            r'^(and|also|what about|how about|another)',
            r'^(tell me more|elaborate|continue)',
            r'^(what else|anything else)',
        ],
    }
    
    # Keyword weights for scoring
    KEYWORDS = {
        Intent.GREETING: ["hi", "hello", "hey", "greetings", "morning", "afternoon", "evening"],
        Intent.FAREWELL: ["bye", "goodbye", "later", "cya", "night"],
        Intent.GRATITUDE: ["thanks", "thank", "appreciate", "grateful"],
        
        Intent.REQUEST_SEARCH: ["search", "find", "lookup", "google", "latest", "current", "news", "recent"],
        Intent.REQUEST_CALCULATE: ["calculate", "compute", "sum", "add", "subtract", "multiply", "divide", "math"],
        Intent.REQUEST_CREATE: ["write", "create", "generate", "make", "build", "compose", "draft"],
        Intent.REQUEST_EXPLAIN: ["explain", "describe", "elaborate", "clarify", "meaning"],
        
        Intent.QUESTION_FACTUAL: ["what", "who", "where", "when", "which", "how many", "how much"],
        Intent.QUESTION_HOW_TO: ["how to", "how do", "how can", "steps", "guide", "tutorial"],
        Intent.QUESTION_WHY: ["why", "reason", "cause"],
        Intent.QUESTION_OPINION: ["think", "opinion", "believe", "recommend", "better", "best"],
        
        Intent.FEEDBACK_POSITIVE: ["great", "good", "awesome", "perfect", "yes", "correct", "right"],
        Intent.FEEDBACK_NEGATIVE: ["no", "wrong", "incorrect", "bad", "not right"],
    }
    
    def __init__(self):
        # Compile patterns
        self._compiled_patterns = {}
        for intent, patterns in self.PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def classify(self, message: str) -> IntentResult:
        """
        Classify user message intent.
        
        Args:
            message: User message text
            
        Returns:
            IntentResult with primary intent and confidence
        """
        message = message.strip()
        message_lower = message.lower()
        
        # Initialize scores
        scores = {intent: 0.0 for intent in Intent}
        features = {}
        
        # 1. Pattern matching (highest weight)
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message_lower):
                    scores[intent] += 0.5
                    break
        
        # 2. Keyword matching
        words = set(message_lower.split())
        for intent, keywords in self.KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in message_lower)
            if matches > 0:
                scores[intent] += 0.3 * min(matches, 3) / 3
        
        # 3. Structural features
        features["is_question"] = message.endswith("?") or any(
            message_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "is", "are", "can", "could", "would", "should", "do", "does"]
        )
        features["is_short"] = len(message.split()) <= 3
        features["is_command"] = any(
            message_lower.startswith(w) for w in ["please", "can you", "could you", "would you", "help me"]
        )
        features["has_numbers"] = bool(re.search(r'\d+', message))
        features["has_math_ops"] = bool(re.search(r'[\+\-\*\/\=]', message))
        
        # Apply feature boosts
        if features["is_question"]:
            for intent in [Intent.QUESTION_FACTUAL, Intent.QUESTION_HOW_TO, Intent.QUESTION_WHY, Intent.QUESTION_OPINION]:
                scores[intent] += 0.2
        
        if features["is_short"]:
            for intent in [Intent.GREETING, Intent.FAREWELL, Intent.GRATITUDE, Intent.FEEDBACK_POSITIVE, Intent.FEEDBACK_NEGATIVE]:
                scores[intent] += 0.1
        
        if features["has_math_ops"] or features["has_numbers"]:
            scores[Intent.REQUEST_CALCULATE] += 0.2
        
        # 4. Default boost for common intents
        if max(scores.values()) < 0.2:
            if features["is_question"]:
                scores[Intent.QUESTION_FACTUAL] += 0.3
            else:
                scores[Intent.STATEMENT] += 0.2
                scores[Intent.CHITCHAT] += 0.1
        
        # Normalize and rank
        total = sum(scores.values()) + 1e-6
        normalized = {k: v / total for k, v in scores.items()}
        
        # Sort by score
        ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        
        primary_intent = ranked[0][0]
        confidence = ranked[0][1]
        secondary = [(intent, score) for intent, score in ranked[1:4] if score > 0.1]
        
        return IntentResult(
            primary_intent=primary_intent,
            confidence=min(confidence * 2, 1.0),  # Scale confidence
            secondary_intents=secondary,
            raw_scores=normalized,
            features=features,
        )
    
    def get_action_suggestion(self, intent_result: IntentResult) -> int:
        """
        Suggest DQN action based on intent.
        
        Returns:
            Suggested action ID (0-7)
        """
        intent = intent_result.primary_intent
        
        # Action mappings
        INTENT_TO_ACTION = {
            # Direct response (0)
            Intent.GREETING: 0,
            Intent.FAREWELL: 0,
            Intent.GRATITUDE: 0,
            Intent.FEEDBACK_POSITIVE: 0,
            Intent.FEEDBACK_NEGATIVE: 0,
            Intent.STATEMENT: 0,
            Intent.CHITCHAT: 0,
            
            # Clarify (1)
            Intent.CLARIFICATION: 1,
            Intent.UNKNOWN: 1,
            
            # Use tool (2)
            Intent.REQUEST_CALCULATE: 2,
            
            # Search web (3)
            Intent.REQUEST_SEARCH: 3,
            Intent.QUESTION_FACTUAL: 3,  # Often needs current info
            
            # Break down task (4)
            Intent.REQUEST_CREATE: 4,
            Intent.QUESTION_HOW_TO: 4,
            
            # Reflect and reason (5)
            Intent.QUESTION_WHY: 5,
            Intent.QUESTION_OPINION: 5,
            Intent.REQUEST_EXPLAIN: 5,
            
            # Defer (6) - rarely suggested
            
            # Cognitive process (7)
            Intent.CORRECTION: 7,
            Intent.FOLLOWUP: 7,
        }
        
        return INTENT_TO_ACTION.get(intent, 0)


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing IntentClassifier...")
    
    classifier = IntentClassifier()
    
    test_cases = [
        "Hello!",
        "What's the weather like today?",
        "Search for Python tutorials",
        "Calculate 25 * 4 + 10",
        "Why is the sky blue?",
        "Can you write a poem about cats?",
        "Thanks, that was helpful!",
        "No, that's not what I meant",
        "How do I install Docker?",
        "What do you think about AI?",
        "Tell me more",
        "Bye!",
    ]
    
    print("\nIntent Classification Results:")
    print("-" * 60)
    
    for msg in test_cases:
        result = classifier.classify(msg)
        action = classifier.get_action_suggestion(result)
        
        print(f"\n'{msg}'")
        print(f"  Intent: {result.primary_intent.value} ({result.confidence:.2f})")
        print(f"  Suggested Action: {action}")
        if result.secondary_intents:
            secondary = ", ".join(f"{i.value}:{s:.2f}" for i, s in result.secondary_intents)
            print(f"  Secondary: {secondary}")
    
    print("\n✅ Intent classifier tests passed!")
