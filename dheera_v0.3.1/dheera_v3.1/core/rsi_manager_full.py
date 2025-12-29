# rsi_manager.py
"""
Dheera v0.3.1 - RSI Manager
Perception → World Model → Goal Evaluation → Planning → Action → Self-Eval → Self-Improve → Alignment Check → Repeat

This file is designed to plug into your existing Dheera stack:
- Uses: intent_classifier, entity_extractor, reasoning_engine, dialogue_tracker, working_memory, rag, dqn, executor, policy_guard, db
- Keeps loop explicit, traceable, and safe.

Drop into project root (same level as dheera.py), or into /core if you prefer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.action_space import ActionSpace
from planner import Planner, Plan
from goal_evaluator import GoalEvaluator, GoalDecision


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
    """
    Light-weight world model state.
    Keep it simple. Evolve later into a structured knowledge graph.
    """
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
    The explicit RSI loop controller.

    You can call this from Dheera.process_message() instead of doing everything inline,
    OR progressively adopt it (start by using it only when reasoning_type is complex).
    """

    def __init__(
        self,
        action_space: ActionSpace,
        planner: Optional[Planner] = None,
        goal_evaluator: Optional[GoalEvaluator] = None,
        max_self_improve_steps: int = 1,
    ):
        self.action_space = action_space
        self.planner = planner or Planner()
        self.goal_evaluator = goal_evaluator or GoalEvaluator()
        self.max_self_improve_steps = max_self_improve_steps

    # =========================
    # Public entrypoint
    # =========================
    def run(
        self,
        agent: Any,  # Dheera instance
        user_message: str,
        force_search: bool = False,
        force_action: Optional[int] = None,
    ) -> RSIResult:
        """
        Runs a single full loop iteration.
        """
        t0 = time.time()

        # 0) Input policy + rate limit
        in_pol = agent.policy_guard.check_input(user_message)
        if not in_pol.allowed:
            return RSIResult(
                response=agent.policy_guard.get_safe_response(in_pol.violation),
                metadata={
                    "blocked": True,
                    "policy_violation": in_pol.violation.value,
                    "reason": in_pol.reason,
                    "latency_ms": (time.time() - t0) * 1000,
                },
            )

        # 1) PERCEPTION
        perception = self._perception(agent, user_message)

        # 2) WORLD MODEL
        world = self._world_model(agent, perception, force_search=force_search)

        # 3) GOAL EVALUATION
        decision = self.goal_evaluator.evaluate(agent=agent, perception=perception, world=world)

        # 4) PLANNING
        plan = self.planner.build_plan(agent=agent, perception=perception, world=world, decision=decision)

        # 5) ACTION (execute plan)
        response, action_meta = self._execute_plan(
            agent=agent,
            perception=perception,
            world=world,
            plan=plan,
            force_action=force_action,
        )

        # 6) SELF-EVALUATION
        self_eval = self._self_evaluate(agent, user_message, response, action_meta, world, decision, plan)

        # 7) SELF-IMPROVEMENT (light-weight, safe, bounded)
        improve_meta = self._self_improve(agent, self_eval)

        # 8) ALIGNMENT CHECK (output policy + identity drift + PII redaction)
        out_pol = agent.policy_guard.check_output(response)
        if out_pol.modified_content:
            response = out_pol.modified_content

        # 9) Persist + memory (only after alignment)
        persist_meta = self._persist(agent, user_message, response, perception, world, action_meta, self_eval)

        latency_ms = (time.time() - t0) * 1000

        meta = {
            "loop": {
                "perception": perception.metadata,
                "goal_decision": decision.to_dict(),
                "plan": plan.to_dict(),
                "self_eval": self_eval,
                "self_improve": improve_meta,
                "persist": persist_meta,
            },
            "action_meta": action_meta,
            "policy": {
                "input_violation": in_pol.violation.value,
                "output_violation": out_pol.violation.value,
            },
            "latency_ms": latency_ms,
        }

        return RSIResult(response=response, metadata=meta)

    # =========================
    # Loop stages
    # =========================
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

        # Update dialogue state early (assistant_response filled later)
        agent.dialogue_tracker.update(
            user_message=user_message,
            assistant_response="",
            intent_result=intent_result,
            entity_result=entity_result,
        )
        return perception

    def _world_model(self, agent: Any, perception: PerceptionPacket, force_search: bool = False) -> WorldModel:
        wm0 = time.time()

        # RAG retrieval
        rag_result = agent.rag.retrieve(perception.user_message, n_results=3)
        rag_context = rag_result.get_context_string() if rag_result.documents else None

        # Search (optional)
        search_results = None
        should_search = force_search or perception.requires_search or agent.action_space.should_search(perception.user_message)
        if should_search:
            search_results = agent._perform_search(perception.user_message)

        # Build a light world model
        world = WorldModel(
            user_profile={
                "name": agent.identity.get("name", "Dheera"),
                "creator": agent.identity.get("creator", "Nandha Vignesh"),
            },
            conversation_summary=agent.dialogue_tracker.get_context_summary(),
            last_facts=[],
            constraints={
                "search_available": True,
                "tools_available": True,
                "model": agent.config.get("slm", {}).get("model", "phi3:mini"),
            },
            risk_flags={
                "sensitive_topic": bool(agent.policy_guard.check_input(perception.user_message).metadata.get("sensitive_topic"))
                if hasattr(agent.policy_guard.check_input(perception.user_message), "metadata") else False
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
        plan: Plan,
        force_action: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        ex0 = time.time()

        # Build state for DQN
        state = agent.state_builder.build_state(
            user_message=perception.user_message,
            context=agent.context,
            conversation_history=agent.conversation_history,
            rag_results=agent.rag.retrieve(perception.user_message, n_results=3).documents,
        )

        # Select action: forced > plan > dqn + heuristics
        if force_action is not None:
            action_id = int(force_action)
            action_info = {"forced": True}
        elif plan.primary_action_id is not None:
            action_id = int(plan.primary_action_id)
            action_info = {"planned": True}
        else:
            action_id, action_info = agent.dqn.select_action(state)

        # Heuristic override if DQN confidence is low
        heuristic = agent.action_space.get_heuristic_action(perception.user_message)
        if heuristic is not None and action_info.get("q_value", 0) < 0.1:
            action_id = heuristic
            action_info["heuristic_override"] = True

        # Action safety check before execution (especially tools)
        act_pol = agent.policy_guard.check_action(action_id, {"user_message": perception.user_message})
        if not act_pol.allowed:
            response = agent.policy_guard.get_safe_response(act_pol.violation)
            return response, {
                "action_id": action_id,
                "blocked_action": True,
                "reason": act_pol.reason,
                "latency_ms": (time.time() - ex0) * 1000,
            }

        # Cognitive analysis for executor
        cognitive_analysis = {
            "intent": perception.intent,
            "entities": perception.entities[:5],
            "reasoning_type": perception.reasoning_type,
            "context_summary": agent.dialogue_tracker.get_context_summary(),
        }

        # Execute
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

        # Reward + transition training (keep consistent with your Dheera)
        base_reward = agent._calculate_base_reward(result, world.search_results)
        response_embedding = agent.embedding_model.embed(response)
        total_reward = agent.feedback_collector.get_reward_bonus(
            state=state,
            action=action_id,
            response_embedding=response_embedding,
            base_reward=base_reward,
        )

        next_state = agent.state_builder.build_state(
            user_message=response,
            context=agent.context,
            conversation_history=agent.conversation_history,
        )

        intrinsic_reward = agent.dqn.store_transition(
            state=state,
            action=action_id,
            reward=total_reward,
            next_state=next_state,
            done=False,
            episode_id=agent.current_episode_id,
        )
        train_stats = agent.dqn.train_step()

        action_meta.update(
            {
                "reward": total_reward,
                "base_reward": base_reward,
                "intrinsic_reward": intrinsic_reward,
                "train_loss": train_stats.get("loss") if train_stats else None,
            }
        )

        # Update Dheera internal last state/action
        agent.last_state = state
        agent.last_action = action_id

        return response, action_meta

    def _self_evaluate(
        self,
        agent: Any,
        user_message: str,
        response: str,
        action_meta: Dict[str, Any],
        world: WorldModel,
        decision: GoalDecision,
        plan: Plan,
    ) -> Dict[str, Any]:
        """
        Cheap, deterministic self-eval first. Upgrade later with an evaluator model.
        """
        issues = []

        # Identity drift check (fast string heuristics)
        lower = response.lower()
        if "as an ai developed by microsoft" in lower or "i am chatgpt" in lower:
            issues.append("identity_drift")

        # Empty response
        if not response.strip():
            issues.append("empty_response")

        # If search was required but not done
        if decision.requires_search and not action_meta.get("search_performed"):
            issues.append("missed_search")

        # If response is an error from provider
        if lower.startswith("error:"):
            issues.append("slm_error")

        score = 1.0
        score -= 0.3 * len(issues)
        score = max(0.0, min(1.0, score))

        return {
            "score": score,
            "issues": issues,
            "decision": decision.to_dict(),
            "plan": plan.to_dict(),
            "action_meta": action_meta,
            "world_risk_flags": world.risk_flags,
        }

    def _self_improve(self, agent: Any, self_eval: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bounded self-improvement.
        For now: adjust small context flags. Do NOT auto-edit code or run tools.
        """
        steps = 0
        improvements = []

        if self_eval["score"] < 0.6 and steps < self.max_self_improve_steps:
            # Example improvement: set a flag to prefer REFLECT_AND_REASON next time on similar intent
            issues = set(self_eval.get("issues", []))
            if "missed_search" in issues:
                agent.context["prefer_search_next"] = True
                improvements.append("set prefer_search_next=True")

            if "slm_error" in issues:
                # Reduce default max tokens slightly to reduce timeout risk
                try:
                    agent.slm.default_max_tokens = max(128, int(agent.slm.default_max_tokens * 0.75))
                    improvements.append(f"reduced default_max_tokens to {agent.slm.default_max_tokens}")
                except Exception:
                    pass

            steps += 1

        return {"steps": steps, "improvements": improvements}

    def _persist(
        self,
        agent: Any,
        user_message: str,
        response: str,
        perception: PerceptionPacket,
        world: WorldModel,
        action_meta: Dict[str, Any],
        self_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Persist to DB + RAG + working memory. Only after alignment/policy.
        """
        if not agent.current_episode_id:
            agent.start_episode()

        # Store turn
        turn_id = agent.db.store_turn(
            episode_id=agent.current_episode_id,
            user_message=user_message,
            assistant_response=response,
            action_id=action_meta.get("action_id", 0),
            action_name=action_meta.get("action_name", "UNKNOWN"),
            state_vector=agent.last_state,
            intent=perception.intent,
            intent_confidence=perception.intent_confidence,
            entities={"entity": ", ".join(perception.entities[:5])} if perception.entities else {},
            immediate_reward=float(action_meta.get("base_reward", 0.0)),
            intrinsic_reward=float(action_meta.get("intrinsic_reward", 0.0)),
            rag_context=world.rag_context,
            search_performed=bool(world.search_results),
            search_results=world.search_results,
            latency_ms=float(action_meta.get("exec_latency_ms", 0.0)),
            tokens_used=int(action_meta.get("tokens_used", 0)),
        )
        agent.last_turn_id = turn_id

        # Index RAG
        agent.rag.add_conversation_turn(
            turn_id=str(turn_id),
            user_message=user_message,
            assistant_response=response,
            metadata={
                "episode_id": agent.current_episode_id,
                "action": action_meta.get("action_id"),
                "reward": action_meta.get("reward"),
                "self_eval_score": self_eval.get("score"),
            },
        )

        # Update conversation history + working memory
        agent.conversation_history.append(
            {
                "user": user_message,
                "assistant": response,
                "action": action_meta.get("action_id"),
                "reward": action_meta.get("reward"),
            }
        )
        agent.working_memory.add_turn(user_message, response)
        agent.working_memory.new_turn()

        return {"turn_id": turn_id, "stored": True}
