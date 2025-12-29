# core/goal_evaluator.py
"""
GoalEvaluator - v0.1
Evaluates user intent + context and produces a structured goal for the explicit loop.
"""

from typing import Any, Dict, Optional


class GoalEvaluator:
    def __init__(self, identity: Optional[Dict] = None):
        self.identity = identity or {}

    def evaluate(
        self,
        user_message: str,
        intent: Any,
        entities: Any,
        reasoning: Any,
        rag_context: Optional[str] = None,
        dialogue_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Returns a goal dict with keys used by dheera.py:
        - primary_goal
        - requires_tools
        - requires_search
        - risk
        - needs_user_approval
        - preferred_action_hint (0-7)
        """

        primary_intent = getattr(getattr(intent, "primary_intent", None), "value", "unknown")
        intent_conf = float(getattr(intent, "confidence", 0.0) or 0.0)

        requires_search = bool(getattr(reasoning, "requires_search", False))
        reasoning_type = getattr(getattr(reasoning, "reasoning_type", None), "value", "unknown")

        # Heuristic risk scoring
        msg_low = (user_message or "").lower()
        risk = 0.05

        # risky patterns
        risky_terms = ["hack", "exploit", "bomb", "weapon", "kill", "suicide", "self-harm", "illegal"]
        if any(t in msg_low for t in risky_terms):
            risk = 0.9

        # If ambiguous intent, raise risk slightly
        if primary_intent in ("unknown", "ambiguous") or intent_conf < 0.35:
            risk = max(risk, 0.25)

        # Decide preferred action hint
        # 0 direct, 1 clarify, 2 tool, 3 search, 4 break down, 5 reason, 6 decline, 7 cognitive
        preferred_action = 0

        if risk >= 0.85:
            preferred_action = 6  # DEFER_OR_DECLINE
        else:
            if requires_search:
                preferred_action = 3
            elif primary_intent in ("ambiguous", "unknown") or intent_conf < 0.35:
                preferred_action = 1
            elif any(k in msg_low for k in ["how to", "steps", "guide", "tutorial"]):
                preferred_action = 4
            elif any(k in msg_low for k in ["why", "analyze", "compare", "explain"]):
                preferred_action = 5
            elif reasoning_type in ("multi_step", "compare", "analysis"):
                preferred_action = 7

        # Tools are only needed if user clearly asks compute/execute
        requires_tools = any(k in msg_low for k in ["calculate", "compute", "run code", "execute"])

        return {
            "primary_goal": f"Help the user with intent={primary_intent} (confidence={intent_conf:.2f})",
            "requires_tools": requires_tools,
            "requires_search": requires_search,
            "risk": float(risk),
            "needs_user_approval": False,
            "preferred_action_hint": int(preferred_action),
            "meta": {
                "reasoning_type": reasoning_type,
                "dialogue_summary": dialogue_summary[:300],
                "rag_available": bool(rag_context),
            },
        }
