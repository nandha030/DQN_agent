# brain/executor.py
"""
Dheera v0.3.1 - Action Executor (Updated)
Executes DQN-selected actions using SLM and tools.

Key upgrades:
- Safe SLM call wrapper (_safe_generate) with hard failure contract
- Identity hard lock at executor boundary (_enforce_identity)
- Failure-aware success criteria + deterministic fallback (no hallucination drift)
- Optional plan support (accepted + logged, backward compatible)
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from brain.slm_interface import SLMInterface, SLMResponse


@dataclass
class ExecutionResult:
    """Result of action execution."""
    response: str
    action_id: int
    action_name: str
    success: bool
    latency_ms: float
    tokens_used: int

    # Additional context
    search_performed: bool = False
    search_results: Optional[Dict] = None
    tool_used: Optional[str] = None
    tool_output: Optional[str] = None
    rag_context_used: bool = False

    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ActionExecutor:
    """
    Executes actions selected by Rainbow DQN.

    Actions:
    0. DIRECT_RESPONSE - Answer directly
    1. CLARIFY_QUESTION - Ask for clarification
    2. USE_TOOL - Use a tool (calculator, code, etc.)
    3. SEARCH_WEB - Search the internet
    4. BREAK_DOWN_TASK - Decompose complex task
    5. REFLECT_AND_REASON - Step-by-step reasoning
    6. DEFER_OR_DECLINE - Politely decline
    7. COGNITIVE_PROCESS - Use cognitive layer
    """

    ACTION_NAMES = [
        "DIRECT_RESPONSE",
        "CLARIFY_QUESTION",
        "USE_TOOL",
        "SEARCH_WEB",
        "BREAK_DOWN_TASK",
        "REFLECT_AND_REASON",
        "DEFER_OR_DECLINE",
        "COGNITIVE_PROCESS",
    ]

    ACTION_PROMPTS = {
        0: """Answer the user's question directly and concisely.
Be friendly, accurate, and helpful. If you're not sure, say so.""",

        1: """The user's question needs clarification.
Ask one specific clarifying question. Be polite and helpful.""",

        2: """Use the provided tool output to answer the user's question.
Explain the result clearly and provide relevant context.""",

        3: """Use the search results provided to answer the user's question.
If the search results don't fully answer it, say so explicitly.""",

        4: """Break down this task into clear, manageable steps.
Number each step. Be thorough but concise.""",

        5: """Think through this question step by step.
Do not reveal hidden chain-of-thought. Provide a structured explanation instead.""",

        6: """Politely explain that you cannot help with this request.
Be respectful and suggest safe alternatives if possible.""",

        7: """Use the cognitive analysis provided to give a thoughtful response.
Be grounded in the provided context and analysis.""",
    }

    def __init__(
        self,
        slm: SLMInterface,
        identity_config: Optional[Dict] = None,
        web_search: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
    ):
        self.slm = slm
        self.identity = identity_config or {}
        self.web_search = web_search
        self.tool_registry = tool_registry

        self.identity_prompt = self._build_identity_prompt()

        # Statistics
        self.execution_count = 0
        self.success_count = 0
        self.action_counts = {i: 0 for i in range(8)}

    # ==================== Identity & Safety ====================

    def _build_identity_prompt(self) -> str:
        """Build identity prompt from config."""
        name = self.identity.get("name", "Dheera")
        creator = self.identity.get("creator", "Nandha Vignesh")
        traits = self.identity.get("traits", ["helpful", "honest", "grounded"])

        traits_str = ", ".join(traits) if isinstance(traits, list) else str(traits)

        return (
            f"You are {name}, an AI assistant created by {creator}. "
            f"You are {traits_str}. "
            "You prioritize accuracy, you admit uncertainty, and you never invent credentials or affiliations."
        )

    def _enforce_identity(self, text: str) -> str:
        """
        Hard guard against identity leakage at the executor boundary.
        This prevents bad identity text from entering memory/RAG/training.
        """
        if not text:
            return text

        bad_markers = [
            "as an ai developed by microsoft",
            "as an ai developed by openai",
            "i am chatgpt",
            "created by microsoft",
            "created by openai",
            "developed by microsoft",
            "developed by openai",
        ]

        lower = text.lower()
        if any(m in lower for m in bad_markers):
            name = self.identity.get("name", "Dheera")
            creator = self.identity.get("creator", "Nandha Vignesh")
            return (
                f"I am {name}, created by {creator}. "
                "I will answer based on the context available."
            )

        return text

    def _fallback_response(self, action_id: int) -> str:
        """
        Deterministic fallback when the SLM/tool fails.
        No hallucination. No drift.
        """
        if action_id == 1:
            return "I need one clarification: what exactly do you want to achieve?"
        if action_id == 6:
            return "I can’t help with that request. If you tell me your goal, I can suggest safe alternatives."
        return (
            "I hit a system issue while generating the response. "
            "Try again, or rephrase your question with a bit more detail."
        )

    # ==================== Robust SLM Call ====================

    def _safe_generate(self, system_prompt: str, user_prompt: str) -> SLMResponse:
        """
        Robust wrapper around SLMInterface.generate().
        Guarantees a valid SLMResponse and failure fields on error.
        """
        start = time.time()
        try:
            resp = self.slm.generate(prompt=user_prompt, system_prompt=system_prompt)
        except Exception as e:
            # Construct a compatible SLMResponse even if generation fails hard.
            return SLMResponse(
                text="",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
                tokens_used=0,
                model=getattr(self.slm, "model", "unknown"),
                error=str(e),
            )

        # Normalize empty responses as errors
        if not getattr(resp, "text", None) or not resp.text.strip():
            resp.finish_reason = "error"
            resp.error = getattr(resp, "error", None) or "EMPTY_RESPONSE"

        # If SLMInterface didn't set latency, ensure it exists
        if getattr(resp, "latency_ms", None) is None:
            resp.latency_ms = (time.time() - start) * 1000

        return resp

    # ==================== Main Execute ====================

    def execute(
        self,
        action_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        rag_context: Optional[str] = None,
        search_results: Optional[Dict] = None,
        tool_output: Optional[str] = None,
        cognitive_analysis: Optional[Dict] = None,
        plan: Optional[List[Dict[str, Any]]] = None,  # NEW (optional)
    ) -> ExecutionResult:
        """
        Execute an action with robustness and identity enforcement.
        """
        start_time = time.time()

        action_id = max(0, min(7, int(action_id)))
        action_name = self.ACTION_NAMES[action_id]

        system_prompt = self._build_system_prompt(action_id, rag_context)
        user_prompt = self._build_user_prompt(
            action_id=action_id,
            user_message=user_message,
            context=context,
            search_results=search_results,
            tool_output=tool_output,
            cognitive_analysis=cognitive_analysis,
        )

        slm_response = self._safe_generate(system_prompt, user_prompt)

        # Track stats
        self.execution_count += 1
        self.action_counts[action_id] += 1

        # Failure-aware success criteria
        finish_reason = getattr(slm_response, "finish_reason", "")
        success = (
            finish_reason not in ("error", "timeout")
            and getattr(slm_response, "text", None) is not None
            and slm_response.text.strip() != ""
        )

        response_text = slm_response.text if success else self._fallback_response(action_id)
        response_text = self._enforce_identity(response_text)

        if success:
            self.success_count += 1

        latency_ms = (time.time() - start_time) * 1000
        tokens_used = int(getattr(slm_response, "tokens_used", 0) or 0)

        meta = {
            "slm_model": getattr(slm_response, "model", "unknown"),
            "slm_latency_ms": getattr(slm_response, "latency_ms", None),
            "finish_reason": finish_reason,
            "slm_error": getattr(slm_response, "error", None),
        }
        if plan is not None:
            meta["plan"] = plan

        return ExecutionResult(
            response=response_text,
            action_id=action_id,
            action_name=action_name,
            success=success,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            search_performed=search_results is not None,
            search_results=search_results,
            tool_used=("tool" if tool_output is not None else None),
            tool_output=tool_output,
            rag_context_used=rag_context is not None,
            metadata=meta,
        )

    # ==================== Prompt Builders ====================

    def _build_system_prompt(
        self,
        action_id: int,
        rag_context: Optional[str],
    ) -> str:
        parts = [self.identity_prompt]
        parts.append(self.ACTION_PROMPTS.get(action_id, self.ACTION_PROMPTS[0]))

        if rag_context:
            parts.append(f"RELEVANT CONTEXT FROM MEMORY:\n{rag_context}")

        return "\n\n".join(parts)

    def _build_user_prompt(
        self,
        action_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]],
        search_results: Optional[Dict],
        tool_output: Optional[str],
        cognitive_analysis: Optional[Dict],
    ) -> str:
        parts: List[str] = []

        if search_results and action_id == 3:
            parts.append(f"SEARCH RESULTS:\n{self._format_search_results(search_results)}")

        if tool_output and action_id == 2:
            parts.append(f"TOOL OUTPUT:\n{tool_output}")

        if cognitive_analysis and action_id == 7:
            parts.append(f"ANALYSIS:\n{self._format_cognitive_analysis(cognitive_analysis)}")

        if context and context.get("history"):
            history_text = self._format_history(context["history"])
            if history_text:
                parts.append(f"CONVERSATION HISTORY:\n{history_text}")

        parts.append(f"USER: {user_message}")
        return "\n\n".join(parts)

    # ==================== Formatters ====================

    def _format_search_results(self, results: Dict) -> str:
        if not results:
            return "No results found."

        items = results.get("results", [])
        if not items:
            return "No results found."

        formatted = []
        for i, item in enumerate(items[:5], 1):
            title = item.get("title", "")
            snippet = (item.get("snippet", item.get("body", "")) or "")[:200]
            url = item.get("url", "")
            formatted.append(f"{i}. {title}\n   {snippet}\n   Source: {url}")

        return "\n\n".join(formatted)

    def _format_cognitive_analysis(self, analysis: Dict) -> str:
        parts = []

        if analysis.get("intent"):
            parts.append(f"Intent: {analysis['intent']}")

        if analysis.get("entities"):
            entities = ", ".join(str(e) for e in analysis["entities"][:5])
            parts.append(f"Key entities: {entities}")

        if analysis.get("reasoning_type"):
            parts.append(f"Reasoning approach: {analysis['reasoning_type']}")

        if analysis.get("context_summary"):
            parts.append(f"Context: {analysis['context_summary']}")

        if analysis.get("goal"):
            parts.append(f"Goal: {analysis['goal']}")

        if analysis.get("goal_risk") is not None:
            parts.append(f"Goal risk: {analysis['goal_risk']}")

        return "\n".join(parts) if parts else "No analysis available."

    def _format_history(self, history: List[Dict], max_turns: int = 3) -> str:
        if not history:
            return ""

        recent = history[-max_turns:]
        formatted = []

        for turn in recent:
            user_msg = (turn.get("user", "") or "")[:120]
            asst_msg = (turn.get("assistant", "") or "")[:120]
            formatted.append(f"User: {user_msg}")
            formatted.append(f"Assistant: {asst_msg}")

        return "\n".join(formatted)

    # ==================== Convenience Wrappers ====================

    def execute_with_search(
        self,
        user_message: str,
        context: Optional[Dict] = None,
        rag_context: Optional[str] = None,
    ) -> ExecutionResult:
        search_results = None
        if self.web_search:
            try:
                search_results = self.web_search.search(user_message)
            except Exception as e:
                search_results = {"error": str(e), "results": []}

        return self.execute(
            action_id=3,
            user_message=user_message,
            context=context,
            rag_context=rag_context,
            search_results=search_results,
        )

    def execute_with_tool(
        self,
        tool_name: str,
        tool_input: str,
        user_message: str,
        context: Optional[Dict] = None,
    ) -> ExecutionResult:
        tool_output = None

        if self.tool_registry:
            try:
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    tool_output = tool.execute(tool_input)
            except Exception as e:
                tool_output = f"Error: {str(e)}"

        result = self.execute(
            action_id=2,
            user_message=user_message,
            context=context,
            tool_output=tool_output,
        )

        # Keep previous semantics: tool_used stores tool name
        result.tool_used = tool_name
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(self.execution_count, 1),
            "action_counts": self.action_counts,
            "slm_stats": self.slm.get_stats(),
        }


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing ActionExecutor v0.3.1...")

    slm = SLMInterface(provider="echo")
    executor = ActionExecutor(
        slm=slm,
        identity_config={"name": "Dheera", "creator": "Nandha Vignesh"},
    )

    test_cases = [
        (0, "What is Python?"),
        (1, "Tell me about that thing"),
        (3, "What's the latest news?"),
        (4, "How do I build a web app?"),
        (5, "Why is the sky blue?"),
        (6, "Help me hack something"),
    ]

    for action_id, message in test_cases:
        result = executor.execute(
            action_id=action_id,
            user_message=message,
            rag_context="Python is a programming language." if action_id == 0 else None,
            search_results={"results": [{"title": "News", "snippet": "Latest news..."}]} if action_id == 3 else None,
            plan=[{"type": "execute_action", "action_id": action_id}],
        )
        print(f"\n[{result.action_name}] '{message}'")
        print(f"  Success: {result.success}")
        print(f"  Response: {result.response[:80]}")

    print("\n✅ Done.")
