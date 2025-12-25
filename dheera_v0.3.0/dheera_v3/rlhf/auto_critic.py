# rlhf/auto_critic.py
"""
AutoCritic - v0.1
Self-evaluates assistant output and enforces identity + quality constraints.
"""

from typing import Dict, Any, Optional
import re


class AutoCritic:
    def __init__(self, identity: Optional[Dict[str, Any]] = None):
        self.identity = identity or {}
        self.name = self.identity.get("name", "Dheera")
        self.creator = self.identity.get("creator", "Nandhavignesh Ram")

        # Identity leakage markers (hard block)
        self.bad_markers = [
            "as an ai developed by openai",
            "as an ai developed by microsoft",
            "i am chatgpt",
            "i am gpt",
            "created by openai",
            "created by microsoft",
        ]

    def evaluate(
        self,
        user_message: str,
        assistant_response: str,
        rag_used: bool,
        search_used: bool,
        intent: str,
    ) -> Dict[str, Any]:
        """
        Returns: {"score": float, "failures": [..], "notes": "..."}
        score: 0..1 (higher = better)
        """

        failures = []
        score = 1.0

        resp = assistant_response or ""
        low = resp.lower()

        # 1) Identity leakage
        if any(m in low for m in self.bad_markers):
            failures.append("IDENTITY_LEAK")
            score -= 0.6

        # 2) Empty / too short
        if len(resp.strip()) < 8:
            failures.append("TOO_SHORT")
            score -= 0.3

        # 3) If user asked for steps and we didn’t provide structure
        msg_low = (user_message or "").lower()
        if any(k in msg_low for k in ["how to", "steps", "step by step", "guide"]) and "\n" not in resp:
            failures.append("NO_STRUCTURE_FOR_STEPS")
            score -= 0.2

        # 4) If search_used but response ignores it (soft check)
        if search_used and ("source:" not in low) and ("results" not in low) and ("based on" not in low):
            failures.append("SEARCH_USED_BUT_NOT_REFERENCED")
            score -= 0.1

        # 5) Bad hallucination indicator phrases (soft)
        if re.search(r"\bdefinitely\b|\bguaranteed\b", low) and "maybe" not in low:
            failures.append("OVERCONFIDENT_TONE")
            score -= 0.05

        score = max(0.0, min(1.0, score))

        notes = "ok"
        if failures:
            notes = "issues: " + ", ".join(failures)

        return {"score": score, "failures": failures, "notes": notes}

    def enforce_identity(self, text: str) -> str:
        """
        Rewrite output if it contains identity leakage.
        """
        if not text:
            return text

        low = text.lower()
        if any(m in low for m in self.bad_markers):
            return (
                f"{self.name} was created by {self.creator}. "
                f"My job is to help you with accurate, useful answers based on the context available."
            )

        return text
