# core/rsi_manager.py
"""
Dheera v0.3.1 - RSI Manager (Hardened, config-driven, dheera.py compatible)

Purpose:
- Provide bounded self-improvement hooks + persistent "micro-rules" memory
- Provide small "nudges" text injected into generation context (RAG prompt add-on)
- Optional full RSI loop runner (run()), disabled by default
- Safe by default: no auto code edits, no shell execution, no tool calls by itself

Public methods used by Dheera:
- maybe_improve(eval_report, intent, last_action)
- get_memory_context()  -> small text injected into generation context (RAG prompt add-on)

Recommended integration improvement (optional, but best):
- Use build_rag_context(rag_context_raw) to avoid RAG poisoning and keep raw retrieval separate.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Optional imports for full-loop run()
try:
    from core.action_space import ActionSpace
except Exception:
    ActionSpace = None  # type: ignore

try:
    from cognitive.planner import Planner  # type: ignore
except Exception:
    Planner = None  # type: ignore

try:
    from core.goal_evaluator import GoalEvaluator  # type: ignore
except Exception:
    GoalEvaluator = None  # type: ignore


# -----------------------------
# Dataclasses (for optional run)
# -----------------------------
@dataclass
class PerceptionPacket:
    user_message: str
    intent: str
    intent_confidence: float
    entities: List[str]
    reasoning_type: str
    requires_search: bool
    timestamp_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldModel:
    user_profile: Dict[str, Any] = field(default_factory=dict)
    conversation_summary: str = ""
    last_facts: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    risk_flags: Dict[str, Any] = field(default_factory=dict)
    rag_context: Optional[str] = None
    search_results: Optional[Dict[str, Any]] = None


@dataclass
class RSIResult:
    response: str
    metadata: Dict[str, Any]


class RSIManager:
    """
    RSIManager in this repo is NOT a second brain.
    It is a bounded "self-improvement + memory-of-mistakes" module.

    Designed to be called from Dheera.process_message():

      self.rsi.maybe_improve(eval_report, intent, last_action)
      rsi_ctx = self.rsi.get_memory_context()

    You may also use run() later, but only if you make RSIManager the orchestrator.
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        config_path: Optional[str] = None,
        evolution_dir: str = "./evolution",
        max_rules: int = 25,
        max_daily_events: int = 2000,
        max_event_file_mb: int = 25,
        enabled: bool = True,
        enable_full_loop: bool = False,
        action_space: Optional[Any] = None,
        planner: Optional[Any] = None,
        goal_evaluator: Optional[Any] = None,
        max_self_improve_steps: int = 1,
    ):
        # Integration points
        self.db = db
        self.config_path = config_path

        # Feature gates
        self.enabled = bool(enabled)
        self.enable_full_loop = bool(enable_full_loop)

        # Persistence for evolution and micro-rules
        self.evolution_dir = evolution_dir
        os.makedirs(self.evolution_dir, exist_ok=True)
        self.rules_path = os.path.join(self.evolution_dir, "rsi_rules.json")
        self.events_path = os.path.join(self.evolution_dir, "rsi_events.jsonl")

        # Limits
        self.max_rules = int(max_rules)
        self.max_daily_events = int(max_daily_events)
        self.max_event_file_mb = int(max_event_file_mb)

        # Runtime state
        self._rules: Dict[str, Any] = self._load_rules()
        self._events_written_today = 0
        self._last_day_key = self._day_key()

        # Optional full-loop components (not required by Dheera)
        self.action_space = action_space or (ActionSpace() if (ActionSpace and self.enable_full_loop) else None)
        self.planner = planner or (Planner() if (Planner and self.enable_full_loop) else None)
        self.goal_evaluator = goal_evaluator or (
            GoalEvaluator(identity={}) if (GoalEvaluator and self.enable_full_loop) else None
        )
        self.max_self_improve_steps = int(max_self_improve_steps)

    # ==========================================================
    # Public: called by Dheera.process_message()
    # ==========================================================
    def maybe_improve(self, eval_report: Dict[str, Any], intent: str, last_action: int) -> Dict[str, Any]:
        """
        Bounded self-improvement based on eval_report from AutoCritic.
        Writes:
        - compact event log (jsonl)
        - updates micro-rules (rsi_rules.json)
        Returns: improvement metadata (for observability)

        Safe: no code edits, no tool calls, no shell commands.
        """
        if not self.enabled:
            return {"enabled": False, "event_logged": False, "improvements": [], "rules_snapshot": self._public_rules_snapshot()}

        self._roll_day_if_needed()
        self._maybe_rotate_events_file()

        score = eval_report.get("score", None)

        # Normalize failure keys coming from different evaluators
        failures = eval_report.get("failures", None)
        if failures is None:
            failures = eval_report.get("issues", []) or []
        if not isinstance(failures, list):
            failures = [str(failures)]
        failures = [self._normalize_failure(str(x)) for x in failures if str(x).strip()]
        failures = failures[:10]

        event = {
            "ts_ms": int(time.time() * 1000),
            "day": self._day_key(),
            "intent": str(intent),
            "last_action": int(last_action),
            "score": score,
            "failures": failures,
        }

        wrote = self._append_event(event)

        improvements: List[str] = []

        # Track counts
        if failures:
            for f in failures:
                self._rules["failure_counts"][f] = int(self._rules["failure_counts"].get(f, 0)) + 1

        # --- deterministic rule synthesis ---
        # 1) Missed search (support multiple keys)
        missed_search_keys = {"missed_search", "NO_SEARCH", "MISSING_SEARCH"}
        if any(f in missed_search_keys for f in failures) or self._rules["failure_counts"].get("missed_search", 0) >= 2:
            self._rules["nudges"]["prefer_search"] = True
            improvements.append("rule: prefer_search=True (missed_search pattern)")

        # 2) Identity drift
        if "identity_drift" in failures or self._rules["failure_counts"].get("identity_drift", 0) >= 1:
            self._rules["nudges"]["strict_identity"] = True
            improvements.append("rule: strict_identity=True")

        # 3) Hallucination / ungrounded
        hallucination_like = {"hallucination", "ungrounded", "fact_error", "missing_citation"}
        if any(self._normalize_failure(f) in hallucination_like for f in failures):
            self._rules["nudges"]["groundedness"] = True
            improvements.append("rule: groundedness=True (hallucination-like failure)")

        # 4) Low score streak -> concise
        if isinstance(score, (int, float)):
            if score < 0.6:
                self._rules["low_score_streak"] = int(self._rules.get("low_score_streak", 0)) + 1
            else:
                self._rules["low_score_streak"] = 0

            if self._rules.get("low_score_streak", 0) >= 2:
                self._rules["nudges"]["be_concise"] = True
                improvements.append("rule: be_concise=True (low_score streak)")

        # Keep rules bounded and persist
        self._shrink_rules_if_needed()
        self._save_rules()

        return {
            "enabled": True,
            "event_logged": wrote,
            "improvements": improvements,
            "rules_snapshot": self._public_rules_snapshot(),
        }

    def get_memory_context(self) -> str:
        """
        Returns a compact text blob to inject into RAG context.
        This is how RSI actually changes Dheera behavior.
        """
        if not self.enabled:
            return ""

        nudges = self._rules.get("nudges", {}) or {}
        failure_counts = self._rules.get("failure_counts", {}) or {}

        lines: List[str] = []
        lines.append("RSI_MICRO_RULES (apply silently):")

        if nudges.get("prefer_search"):
            lines.append("- If query is factual/uncertain, prefer SEARCH before answering.")
        if nudges.get("groundedness"):
            lines.append("- Avoid guessing. State uncertainty. Prefer evidence from memory/RAG/search.")
        if nudges.get("be_concise"):
            lines.append("- Keep responses short, practical, and direct.")
        if nudges.get("strict_identity"):
            lines.append("- Never claim OpenAI/Microsoft/ChatGPT identity. Use configured identity only.")

        # Include top 3 recurring failures as a reminder
        if failure_counts:
            top = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if top:
                lines.append("- Recent recurring issues to avoid: " + ", ".join([f"{k}({v})" for k, v in top]))

        return "\n".join(lines).strip()

    def build_rag_context(self, rag_context_raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        BEST PRACTICE helper:
        Returns (combined_context, rsi_ctx) while keeping raw retrieval separate.

        Use like:
          rag_context_raw = rag_context
          rag_context, rsi_ctx = self.rsi.build_rag_context(rag_context_raw)

        This prevents "RSI memory" from being stored as if it came from retrieval.
        """
        if not self.enabled:
            return rag_context_raw, None

        rsi_ctx = ""
        try:
            rsi_ctx = self.get_memory_context() or ""
        except Exception:
            rsi_ctx = ""

        if not rsi_ctx.strip():
            return rag_context_raw, None

        combined = (rag_context_raw or "").strip()
        if combined:
            combined = combined + "\n\n" + rsi_ctx
        else:
            combined = rsi_ctx

        return combined, rsi_ctx

    # ==========================================================
    # Optional: Full RSI loop runner (only if you adopt it later)
    # ==========================================================
    def run(
        self,
        agent: Any,
        user_message: str,
        force_search: bool = False,
        force_action: Optional[int] = None,
    ) -> RSIResult:
        if not self.enable_full_loop:
            return RSIResult(
                response="RSIManager.run() is disabled. Enable enable_full_loop=True if you refactor Dheera to use it.",
                metadata={"enabled": False},
            )

        t0 = time.time()
        in_pol = agent.policy_guard.check_input(user_message)
        if not in_pol.allowed:
            return RSIResult(
                response=agent.policy_guard.get_safe_response(in_pol.violation),
                metadata={"blocked": True, "reason": in_pol.reason, "latency_ms": (time.time() - t0) * 1000},
            )

        perception = self._perception(agent, user_message)
        world = self._world_model(agent, perception, force_search=force_search)

        decision = None
        if self.goal_evaluator is not None and hasattr(self.goal_evaluator, "evaluate"):
            decision = self.goal_evaluator.evaluate(agent=agent, perception=perception, world=world)

        plan = None
        if self.planner is not None and hasattr(self.planner, "build_plan"):
            plan = self.planner.build_plan(agent=agent, perception=perception, world=world, decision=decision)

        response, action_meta = self._execute_plan(agent, perception, world, plan, force_action=force_action)

        out_pol = agent.policy_guard.check_output(response)
        if out_pol.modified_content:
            response = out_pol.modified_content

        latency_ms = (time.time() - t0) * 1000
        return RSIResult(response=response, metadata={"action_meta": action_meta, "latency_ms": latency_ms})

    # -----------------------------
    # Internal helpers for run()
    # -----------------------------
    def _perception(self, agent: Any, user_message: str) -> PerceptionPacket:
        pr0 = time.time()
        intent_result = agent.intent_classifier.classify(user_message)
        entity_result = agent.entity_extractor.extract(user_message)
        reasoning_result = agent.reasoning_engine.reason(user_message)

        perception = PerceptionPacket(
            user_message=user_message,
            intent=intent_result.primary_intent.value,
            intent_confidence=float(intent_result.confidence),
            entities=[e.text for e in entity_result.entities[:10]],
            reasoning_type=reasoning_result.reasoning_type.value,
            requires_search=bool(getattr(reasoning_result, "requires_search", False)),
            timestamp_ms=time.time() * 1000,
            metadata={
                "intent": intent_result.primary_intent.value,
                "intent_confidence": float(intent_result.confidence),
                "entities": [e.text for e in entity_result.entities[:5]],
                "reasoning_type": reasoning_result.reasoning_type.value,
                "perception_latency_ms": (time.time() - pr0) * 1000,
            },
        )

        agent.dialogue_tracker.update(
            user_message=user_message,
            assistant_response="",
            intent_result=intent_result,
            entity_result=entity_result,
        )
        return perception

    def _world_model(self, agent: Any, perception: PerceptionPacket, force_search: bool = False) -> WorldModel:
        wm0 = time.time()

        rag_result = agent.rag.retrieve(perception.user_message, n_results=3)
        rag_context = rag_result.get_context_string() if rag_result.documents else None

        should_search = force_search or perception.requires_search
        if hasattr(agent, "context") and agent.context.get("prefer_search_next"):
            should_search = True

        if hasattr(agent, "action_space") and hasattr(agent.action_space, "should_search"):
            try:
                should_search = should_search or bool(agent.action_space.should_search(perception.user_message))
            except Exception:
                pass

        search_results = None
        if should_search:
            search_results = agent._perform_search(perception.user_message)

        world = WorldModel(
            user_profile={
                "name": agent.identity.get("name", "Dheera"),
                "creator": agent.identity.get("creator", "Nandha Vignesh"),
            },
            conversation_summary=agent.dialogue_tracker.get_context_summary(),
            constraints={"model": agent.config.get("slm", {}).get("model", "phi3:mini")},
            risk_flags={
                "sensitive_topic": bool(getattr(agent.policy_guard.check_input(perception.user_message), "metadata", {}).get("sensitive_topic", False))
            },
            rag_context=rag_context,
            search_results=search_results,
        )
        world.constraints["world_model_latency_ms"] = (time.time() - wm0) * 1000
        return world

    def _execute_plan(
        self,
        agent: Any,
        perception: PerceptionPacket,
        world: WorldModel,
        plan: Any,
        force_action: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        ex0 = time.time()

        state = agent.state_builder.build_state(
            user_message=perception.user_message,
            context=agent.context,
            conversation_history=agent.conversation_history,
            rag_results=agent.rag.retrieve(perception.user_message, n_results=3).documents,
        )

        if force_action is not None:
            action_id = int(force_action)
            action_info = {"forced": True}
        else:
            primary = getattr(plan, "primary_action_id", None) if plan is not None else None
            if primary is not None:
                action_id = int(primary)
                action_info = {"planned": True}
            else:
                action_id, action_info = agent.dqn.select_action(state)

        heuristic = agent.action_space.get_heuristic_action(perception.user_message)
        if heuristic is not None and action_info.get("q_value", 0) < 0.1:
            action_id = heuristic
            action_info["heuristic_override"] = True

        act_pol = agent.policy_guard.check_action(action_id, {"user_message": perception.user_message})
        if not act_pol.allowed:
            response = agent.policy_guard.get_safe_response(act_pol.violation)
            return response, {"action_id": action_id, "blocked_action": True, "reason": act_pol.reason}

        cognitive_analysis = {
            "intent": perception.intent,
            "entities": perception.entities[:5],
            "reasoning_type": perception.reasoning_type,
            "context_summary": agent.dialogue_tracker.get_context_summary(),
        }

        result = agent.executor.execute(
            action_id=action_id,
            user_message=perception.user_message,
            context=agent.context,
            rag_context=world.rag_context,
            search_results=world.search_results,
            cognitive_analysis=cognitive_analysis,
        )

        response = result.response
        action_meta = {
            "action_id": action_id,
            "action_name": result.action_name,
            "success": result.success,
            "tokens_used": result.tokens_used,
            "slm_latency_ms": result.latency_ms,
            "search_performed": result.search_performed,
            "rag_used": bool(world.rag_context),
            "executor_tool_used": result.tool_used,
            "selection_info": action_info,
            "exec_latency_ms": (time.time() - ex0) * 1000,
        }

        agent.last_state = state
        agent.last_action = action_id
        return response, action_meta

    # ==========================================================
    # Persistence + day rotation + rules maintenance
    # ==========================================================
    def _day_key(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _roll_day_if_needed(self):
        day = self._day_key()
        if day != self._last_day_key:
            self._last_day_key = day
            self._events_written_today = 0

    def _maybe_rotate_events_file(self):
        try:
            if not os.path.exists(self.events_path):
                return
            size_bytes = os.path.getsize(self.events_path)
            size_mb = size_bytes / (1024 * 1024)
            if size_mb < self.max_event_file_mb:
                return
            ts = time.strftime("%Y%m%d_%H%M%S")
            rotated = os.path.join(self.evolution_dir, f"rsi_events_{ts}.jsonl")
            os.replace(self.events_path, rotated)
        except Exception:
            pass

    def _append_event(self, event: Dict[str, Any]) -> bool:
        if self._events_written_today >= self.max_daily_events:
            return False
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events_written_today += 1
            return True
        except Exception:
            return False

    def _normalize_failure(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        upper = s.upper()

        # Map common variants to your canonical keys
        if "MISS" in upper and "SEARCH" in upper:
            return "missed_search"
        if "IDENTITY" in upper and "DRIFT" in upper:
            return "identity_drift"
        if "HALLUCIN" in upper:
            return "hallucination"
        if "UNGROUNDED" in upper:
            return "ungrounded"
        if "FACT" in upper and "ERROR" in upper:
            return "fact_error"
        if "CITATION" in upper and ("MISSING" in upper or "NO" in upper):
            return "missing_citation"

        # Keep original but normalized
        return s.lower()

    def _load_rules(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "nudges": {},
            "failure_counts": {},
            "low_score_streak": 0,
            "updated_ts_ms": int(time.time() * 1000),
        }
        if not os.path.exists(self.rules_path):
            return base
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                base.update(data)
            # Ensure required keys exist
            base.setdefault("nudges", {})
            base.setdefault("failure_counts", {})
            base.setdefault("low_score_streak", 0)
            return base
        except Exception:
            return base

    def _save_rules(self) -> None:
        try:
            self._rules["updated_ts_ms"] = int(time.time() * 1000)
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(self._rules, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _shrink_rules_if_needed(self):
        fc = self._rules.get("failure_counts", {}) or {}
        if len(fc) > self.max_rules:
            top = sorted(fc.items(), key=lambda x: x[1], reverse=True)[: self.max_rules]
            self._rules["failure_counts"] = {k: int(v) for k, v in top}

        nudges = self._rules.get("nudges", {}) or {}
        if len(nudges) > 10:
            keep: Dict[str, Any] = {}
            for k in ["prefer_search", "groundedness", "be_concise", "strict_identity"]:
                if k in nudges:
                    keep[k] = nudges[k]
            for k in nudges:
                if k not in keep and len(keep) < 10:
                    keep[k] = nudges[k]
            self._rules["nudges"] = keep

    def _public_rules_snapshot(self) -> Dict[str, Any]:
        return {
            "nudges": dict(self._rules.get("nudges", {}) or {}),
            "top_failures": sorted(
                (self._rules.get("failure_counts", {}) or {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "low_score_streak": int(self._rules.get("low_score_streak", 0)),
        }
