# cognitive/planner.py
"""
Planner - v0.1
Creates a simple plan object for the explicit loop.
"""

from typing import Any, Dict, List, Optional


class Planner:
    def __init__(self):
        pass

    def build_plan(
        self,
        goal: Dict[str, Any],
        user_message: str,
        rag_context: Optional[str],
        search_results: Optional[Dict[str, Any]],
        cognitive_analysis: Dict[str, Any],
        chosen_action: int,
    ) -> List[Dict[str, Any]]:
        """
        Output: list of plan steps (each is dict)
        Used only for observability + optional future execution engines.
        """

        steps: List[Dict[str, Any]] = []

        # 1) Safety gate
        risk = float(goal.get("risk", 0.0) or 0.0)
        if risk >= 0.85:
            steps.append({"type": "safety_gate", "status": "block_or_decline", "risk": risk})
            steps.append({"type": "execute_action", "action_id": 6})
            return steps

        # 2) Context usage
        if rag_context:
            steps.append({"type": "use_rag_context", "status": "ready", "size": len(rag_context)})

        if search_results:
            steps.append({"type": "use_search_results", "status": "ready", "count": search_results.get("result_count")})

        # 3) If action suggests breakdown, add sub-steps
        msg = (user_message or "").lower()
        if chosen_action == 4 or any(k in msg for k in ["how to", "steps", "guide"]):
            steps.append({"type": "decompose_task", "status": "ready"})
            steps.append({"type": "write_steps", "status": "ready"})

        # 4) If action suggests reasoning
        if chosen_action in (5, 7) or any(k in msg for k in ["compare", "analyze", "explain", "why"]):
            steps.append({"type": "reasoning_pass", "status": "ready", "mode": cognitive_analysis.get("reasoning_type")})

        # 5) Execute
        steps.append({"type": "execute_action", "action_id": int(chosen_action)})

        # 6) Output quality check (auto critic stage is separate, but we plan it)
        steps.append({"type": "self_eval", "status": "ready"})
        steps.append({"type": "alignment_check", "status": "ready"})

        return steps
