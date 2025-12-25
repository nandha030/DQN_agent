# dheera.py
"""
Dheera v0.3.1 - Main Orchestrator (Updated + Fixed)
Explicit closed-loop:
Perception → World Model → Goal Evaluation → Planning → Action
→ Self-Evaluation → Self-Improvement → Alignment Check → Repeat
"""

import os
import yaml
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Core
from core.rainbow_dqn import RainbowDQNAgent
from core.state_builder import StateBuilder
from core.action_space import ActionSpace

# Brain
from brain.slm_interface import SLMInterface
from brain.executor import ActionExecutor, ExecutionResult
from brain.policy import PolicyGuard

# Cognitive
from cognitive.intent_classifier import IntentClassifier
from cognitive.entity_extractor import EntityExtractor
from cognitive.dialogue_state import DialogueStateTracker
from cognitive.working_memory import WorkingMemory
from cognitive.reasoning import ReasoningEngine

# RAG
from rag.embeddings import EmbeddingModel
from rag.retriever import RAGRetriever

# RLHF
from rlhf.reward_model import RewardModel
from rlhf.preference_learner import PreferenceLearner
from rlhf.feedback_collector import FeedbackCollector

# Database
from database.db_manager import DheeraDatabase


# ==================== OPTIONAL LOOP MODULES ====================
try:
    from core.goal_evaluator import GoalEvaluator  # type: ignore
except Exception:
    GoalEvaluator = None

try:
    from cognitive.planner import Planner  # type: ignore
except Exception:
    Planner = None

try:
    from rlhf.auto_critic import AutoCritic  # type: ignore
except Exception:
    AutoCritic = None

try:
    from core.rsi_manager import RSIManager  # type: ignore
except Exception:
    RSIManager = None


class Dheera:
    VERSION = "0.3.1"

    def __init__(
        self,
        config_path: str = "config/dheera_config.yaml",
        identity_path: str = "config/identity.yaml",
        db_path: str = "dheera.db",
        chroma_path: str = "./chroma_db",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.config_path = config_path
        self.identity_path = identity_path
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.checkpoint_dir = checkpoint_dir

        # Load configurations
        self.config = self._load_config(config_path)
        self.identity = self._load_config(identity_path)

        # ✅ Create PolicyGuard ONCE (identity-aware)
        self.policy_guard = PolicyGuard(identity=self.identity)

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("evolution", exist_ok=True)  # optional: for RSI artifacts

        # Initialize all components
        self._init_components()

        # Session state
        self.current_episode_id: Optional[str] = None
        self.context: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_state: Optional[Any] = None
        self.last_action: Optional[int] = None
        self.last_turn_id: Optional[int] = None

        print(f"🧠 Dheera v{self.VERSION} initialized!")
        print(f"   Model: {self.config.get('slm', {}).get('model', 'phi3:mini')}")
        print(f"   Database: {db_path}")

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _init_components(self):
        print("Initializing components...")

        # 1. Database
        self.db = DheeraDatabase(self.db_path)
        print("  ✓ Database")

        # 2. Embedding model
        self.embedding_model = EmbeddingModel(
            model_name=self.config.get("embedding", {}).get("model", "all-MiniLM-L6-v2"),
        )
        print("  ✓ Embedding model")

        # 3. Cognitive components
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.dialogue_tracker = DialogueStateTracker()
        self.working_memory = WorkingMemory()
        self.reasoning_engine = ReasoningEngine()
        print("  ✓ Cognitive layer")

        # 4. State builder
        self.state_builder = StateBuilder(
            embedding_model=self.embedding_model,
            intent_classifier=self.intent_classifier,
            entity_extractor=self.entity_extractor,
            dialogue_tracker=self.dialogue_tracker,
            working_memory=self.working_memory,
        )
        print("  ✓ State builder")

        # 5. Action space
        self.action_space = ActionSpace()
        print("  ✓ Action space")

        # 6. Rainbow DQN
        dqn_config = self.config.get("dqn", {})
        self.dqn = RainbowDQNAgent(
            state_dim=64,
            action_dim=8,
            hidden_dim=dqn_config.get("hidden_dim", 128),
            gamma=dqn_config.get("gamma", 0.99),
            lr=dqn_config.get("lr", 1e-4),
            batch_size=dqn_config.get("batch_size", 64),
            n_step=dqn_config.get("n_step", 3),
            target_update_freq=dqn_config.get("target_update_freq", 1000),
            curiosity_coef=dqn_config.get("curiosity_coef", 0.1),
            db_manager=self.db,
        )
        print("  ✓ Rainbow DQN")

        # 7. RAG
        self.rag = RAGRetriever(
            persist_directory=self.chroma_path,
            embedding_model=self.embedding_model,
        )
        print("  ✓ RAG retriever")

        # 8. SLM interface
        slm_config = self.config.get("slm", {})
        self.slm = SLMInterface(
            provider=slm_config.get("provider", "ollama"),
            model=slm_config.get("model", "phi3:mini"),
            timeout=slm_config.get("timeout", 60),
            default_temperature=slm_config.get("temperature", 0.7),
            default_max_tokens=slm_config.get("max_tokens", 512),
        )
        print("  ✓ SLM interface")

        # 9. Executor
        self.executor = ActionExecutor(
            slm=self.slm,
            identity_config=self.identity,
        )
        print("  ✓ Action executor")

        # ✅ 10. Policy guard already created in __init__
        print("  ✓ Policy guard (identity-aware)")

        # 11. RLHF components
        self.reward_model = RewardModel(
            state_dim=64,
            action_dim=8,
            response_dim=self.embedding_model.embedding_dim,
        )
        self.preference_learner = PreferenceLearner(
            reward_model=self.reward_model,
            db_manager=self.db,
        )
        self.feedback_collector = FeedbackCollector(
            reward_model=self.reward_model,
            preference_learner=self.preference_learner,
            db_manager=self.db,
        )
        print("  ✓ RLHF system")

        # 12. Optional loop modules
        self.goal_evaluator = None
        if GoalEvaluator is not None:
            try:
                self.goal_evaluator = GoalEvaluator(identity=self.identity)
                print("  ✓ Goal evaluator")
            except Exception as e:
                print(f"  ⚠ Goal evaluator unavailable: {e}")

        self.planner = None
        if Planner is not None:
            try:
                self.planner = Planner()
                print("  ✓ Planner")
            except Exception as e:
                print(f"  ⚠ Planner unavailable: {e}")

        self.auto_critic = None
        if AutoCritic is not None:
            try:
                self.auto_critic = AutoCritic(identity=self.identity)
                print("  ✓ Auto-critic")
            except Exception as e:
                print(f"  ⚠ Auto-critic unavailable: {e}")

        self.rsi = None
        if RSIManager is not None:
            try:
                # Your RSIManager signature may differ. Keep it flexible.
                try:
                    self.rsi = RSIManager(db=self.db, config_path=self.config_path)
                except TypeError:
                    self.rsi = RSIManager()
                print("  ✓ RSI manager")
            except Exception as e:
                print(f"  ⚠ RSI manager unavailable: {e}")

    def start_episode(self, user_id: str = "default") -> str:
        self.current_episode_id = self.db.create_episode(user_id=user_id)
        self.conversation_history = []
        self.context = {}
        self.dialogue_tracker.reset()
        self.working_memory.clear()
        return self.current_episode_id

    # ==================== LOOP FALLBACKS ====================

    def _fallback_goal(self, user_message: str, intent: Any, reasoning: Any) -> Dict[str, Any]:
        primary_intent = getattr(getattr(intent, "primary_intent", None), "value", "unknown")
        requires_search = bool(getattr(reasoning, "requires_search", False))
        hint = 0
        if requires_search:
            hint = 3
        elif primary_intent in ("question_factual", "question", "ask"):
            hint = 5
        return {
            "primary_goal": f"Respond to user intent={primary_intent}",
            "requires_tools": False,
            "requires_search": requires_search,
            "risk": 0.1,
            "needs_user_approval": False,
            "preferred_action_hint": hint,
        }

    def _fallback_plan(self, action_id: int, rag_context: Optional[str], search_results: Optional[Dict]) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        if search_results:
            steps.append({"type": "search", "status": "done"})
        if rag_context:
            steps.append({"type": "rag", "status": "done"})
        steps.append({"type": "execute_action", "action_id": action_id})
        return steps

    def _enforce_identity_minimal(self, text: str) -> str:
        bad_markers = [
            "as an ai developed by microsoft",
            "as an ai developed by openai",
            "i am chatgpt",
            "created by microsoft",
            "created by openai",
            "as a large language model",
        ]
        low = (text or "").lower()
        if any(m in low for m in bad_markers):
            creator = self.identity.get("creator", "Nandha Vignesh")
            name = self.identity.get("name", "Dheera")
            return f"I am {name}, created by {creator}. I will answer based on the context available."
        return text

    # ==================== MAIN LOOP ====================

    def process_message(
        self,
        user_message: str,
        force_search: bool = False,
        force_action: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        if not self.current_episode_id:
            self.start_episode()

        # 1) Input policy
        policy_result = self.policy_guard.check_input(user_message)
        if not policy_result.allowed:
            return self.policy_guard.get_safe_response(policy_result.violation), {
                "action": 6,
                "blocked": True,
                "reason": policy_result.reason,
            }

        # 1.5) Feedback path
        if self.feedback_collector.is_feedback(user_message):
            return self._handle_feedback(user_message)

        # 2) Cognitive perception
        intent_result = self.intent_classifier.classify(user_message)
        entity_result = self.entity_extractor.extract(user_message)
        reasoning_result = self.reasoning_engine.reason(user_message)

        # 3) Dialogue update
        self.dialogue_tracker.update(
            user_message=user_message,
            assistant_response="",
            intent_result=intent_result,
            entity_result=entity_result,
        )

        # 4) World model: RAG (⚡ skip for simple greetings/statements)
        skip_rag_intents = {"greeting", "affirmation", "thanks", "farewell"}
        should_skip_rag = (
            hasattr(intent_result, 'primary_intent') and
            hasattr(intent_result.primary_intent, 'value') and
            intent_result.primary_intent.value in skip_rag_intents and
            len(user_message.split()) < 10  # Short messages
        )

        if should_skip_rag:
            rag_result = type('obj', (object,), {'documents': []})()  # Empty result
            rag_context = None
        else:
            rag_result = self.rag.retrieve(user_message, n_results=3)
            rag_context = rag_result.get_context_string() if rag_result.documents else None

        # 4.1) RSI micro-rules injection (optional, safer)
        rag_context_raw = rag_context
        rsi_ctx = None
        if self.rsi is not None and hasattr(self.rsi, "build_rag_context"):
            try:
                rag_context, rsi_ctx = self.rsi.build_rag_context(rag_context_raw)
            except Exception:
                rag_context, rsi_ctx = rag_context_raw, None

        # 5) Build state
        state = self.state_builder.build_state(
            user_message=user_message,
            context=self.context,
            conversation_history=self.conversation_history,
            rag_results=rag_result.documents,
        )

        # 6) Goal evaluation
        if self.goal_evaluator is not None:
            try:
                goal = self.goal_evaluator.evaluate(
                    user_message=user_message,
                    intent=intent_result,
                    entities=entity_result,
                    reasoning=reasoning_result,
                    rag_context=rag_context,
                    dialogue_summary=self.dialogue_tracker.get_context_summary(),
                )
            except Exception:
                goal = self._fallback_goal(user_message, intent_result, reasoning_result)
        else:
            goal = self._fallback_goal(user_message, intent_result, reasoning_result)

        # 7) Select action
        if force_action is not None:
            action_id = int(force_action)
            action_info = {"forced": True}
        else:
            action_id, action_info = self.dqn.select_action(state)

        hint = goal.get("preferred_action_hint")
        if hint is not None and action_info.get("q_value", 0) < 0.15:
            action_id = int(hint)

        heuristic_action = self.action_space.get_heuristic_action(user_message)
        if heuristic_action is not None and action_info.get("q_value", 0) < 0.1:
            action_id = heuristic_action

        # 8) Search if needed
        search_results = None
        if force_search or bool(goal.get("requires_search")) or action_id == 3 or getattr(reasoning_result, "requires_search", False):
            search_results = self._perform_search(user_message)

        # 9) Planning
        cognitive_analysis = {
            "intent": intent_result.primary_intent.value,
            "entities": [e.text for e in entity_result.entities[:5]],
            "reasoning_type": reasoning_result.reasoning_type.value,
            "context_summary": self.dialogue_tracker.get_context_summary(),
            "goal": goal.get("primary_goal"),
            "goal_risk": goal.get("risk", 0.0),
        }

        if self.planner is not None:
            try:
                plan = self.planner.build_plan(
                    goal=goal,
                    user_message=user_message,
                    rag_context=rag_context,
                    search_results=search_results,
                    cognitive_analysis=cognitive_analysis,
                    chosen_action=action_id,
                )
            except Exception:
                plan = self._fallback_plan(action_id, rag_context, search_results)
        else:
            plan = self._fallback_plan(action_id, rag_context, search_results)

        # ✅ 9.5) Action safety check BEFORE execution (critical)
        act_pol = self.policy_guard.check_action(action_id, {"user_message": user_message})
        if not act_pol.allowed:
            return self.policy_guard.get_safe_response(act_pol.violation), {
                "action": 6,
                "blocked_action": True,
                "reason": act_pol.reason,
            }

        # 10) Execute
        try:
            result = self.executor.execute(
                action_id=action_id,
                user_message=user_message,
                context=self.context,
                rag_context=rag_context,
                search_results=search_results,
                cognitive_analysis=cognitive_analysis,
                plan=plan,  # may be ignored
            )
        except TypeError:
            result = self.executor.execute(
                action_id=action_id,
                user_message=user_message,
                context=self.context,
                rag_context=rag_context,
                search_results=search_results,
                cognitive_analysis=cognitive_analysis,
            )

        # 11) Output policy check
        output_policy = self.policy_guard.check_output(result.response)
        if output_policy.modified_content:
            result.response = output_policy.modified_content

        # 12) Self-evaluation
        eval_report: Dict[str, Any] = {"score": None, "failures": []}
        if self.auto_critic is not None:
            try:
                eval_report = self.auto_critic.evaluate(
                    user_message=user_message,
                    assistant_response=result.response,
                    rag_used=(rag_context is not None),
                    search_used=(search_results is not None),
                    intent=intent_result.primary_intent.value,
                )
            except Exception:
                eval_report = {"score": None, "failures": ["AUTO_CRITIC_ERROR"]}

        # 13) Alignment check (identity)
        if self.auto_critic is not None:
            try:
                result.response = self.auto_critic.enforce_identity(result.response)
            except Exception:
                result.response = self._enforce_identity_minimal(result.response)
        else:
            result.response = self._enforce_identity_minimal(result.response)

        # 14) Rewards
        base_reward = self._calculate_base_reward(result, search_results)
        response_embedding = self.embedding_model.embed(result.response)

        total_reward = self.feedback_collector.get_reward_bonus(
            state=state,
            action=action_id,
            response_embedding=response_embedding,
            base_reward=base_reward,
        )

        # 15) Store transition
        next_state = self.state_builder.build_state(
            user_message=result.response,
            context=self.context,
            conversation_history=self.conversation_history,
        )

        intrinsic_reward = self.dqn.store_transition(
            state=state,
            action=action_id,
            reward=total_reward,
            next_state=next_state,
            done=False,
            episode_id=self.current_episode_id,
        )

        # 16) Store DB turn
        self.last_turn_id = self.db.store_turn(
            episode_id=self.current_episode_id,
            user_message=user_message,
            assistant_response=result.response,
            action_id=action_id,
            action_name=result.action_name,
            state_vector=state,
            intent=intent_result.primary_intent.value,
            intent_confidence=intent_result.confidence,
            entities={e.type.value: e.text for e in entity_result.entities[:5]},
            immediate_reward=base_reward,
            intrinsic_reward=intrinsic_reward,
            rag_context=rag_context,
            search_performed=search_results is not None,
            search_results=search_results,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
        )

        # 17) Index into RAG
        self.rag.add_conversation_turn(
            turn_id=str(self.last_turn_id),
            user_message=user_message,
            assistant_response=result.response,
            metadata={
                "action": action_id,
                "reward": total_reward,
                "episode_id": self.current_episode_id,
                "auto_eval_score": eval_report.get("score"),
                "auto_eval_failures": eval_report.get("failures", []),
            },
        )

        # 18) Self-improvement (RSI-lite)
        if self.rsi is not None:
            try:
                self.rsi.maybe_improve(
                    eval_report=eval_report,
                    intent=intent_result.primary_intent.value,
                    last_action=action_id,
                )
            except Exception:
                pass

        # 19) Train step
        train_stats = self.dqn.train_step()

        # 20) Update state
        self.last_state = state
        self.last_action = action_id

        self.conversation_history.append({
            "user": user_message,
            "assistant": result.response,
            "action": action_id,
            "reward": total_reward,
            "auto_eval_score": eval_report.get("score"),
            "auto_eval_failures": eval_report.get("failures", []),
            "goal": goal,
            "plan": plan,
        })

        self.working_memory.add_turn(user_message, result.response)
        self.working_memory.new_turn()

        # 21) Metadata
        total_time_ms = (time.time() - start_time) * 1000

        metadata = {
            "action": action_id,
            "action_name": result.action_name,
            "reward": total_reward,
            "intrinsic_reward": intrinsic_reward,
            "latency_ms": total_time_ms,
            "slm_latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "search_performed": result.search_performed,
            "rag_used": rag_context is not None,
            "intent": intent_result.primary_intent.value,
            "novelty": action_info.get("novelty", 0),
            "train_loss": train_stats.get("loss") if train_stats else None,
            "goal": goal,
            "plan_steps": len(plan) if isinstance(plan, list) else None,
            "auto_eval_score": eval_report.get("score"),
            "auto_eval_failures": eval_report.get("failures", []),
        }

        return result.response, metadata

    def _handle_feedback(self, feedback_str: str) -> Tuple[str, Dict[str, Any]]:
        if self.last_turn_id is None or self.last_state is None:
            return "No previous turn to provide feedback for.", {"error": True}

        last_response = self.conversation_history[-1]["assistant"] if self.conversation_history else ""
        response_embedding = self.embedding_model.embed(last_response)

        feedback_value, info = self.feedback_collector.collect(
            turn_id=self.last_turn_id,
            episode_id=self.current_episode_id,
            user_message=self.conversation_history[-1].get("user", ""),
            assistant_response=last_response,
            action_id=self.last_action,
            state_vector=self.last_state,
            response_embedding=response_embedding,
            feedback_str=feedback_str,
        )

        self.db.update_turn_feedback(self.last_turn_id, feedback_value)

        if feedback_value > 0:
            response = "Thank you for the positive feedback! I'll learn from this. 😊"
        else:
            response = "I understand. I'll try to do better next time. Thanks for the feedback."

        return response, {
            "feedback": feedback_value,
            "preferences_created": info.get("preferences_created", 0),
        }

    def _perform_search(self, query: str) -> Optional[Dict[str, Any]]:
        cached = self.db.get_cached_search(query)
        if cached:
            return cached

        # TODO: integrate real search later
        results = {
            "query": query,
            "results": [
                {"title": "Search result", "snippet": "Search functionality coming soon.", "url": ""},
            ],
            "result_count": 1,
        }

        self.db.cache_search(query, results)
        return results

    def _calculate_base_reward(self, result: ExecutionResult, search_results: Optional[Dict[str, Any]]) -> float:
        reward = 0.0
        if result.success:
            reward += 0.1
        if result.tool_used:
            reward += self.config.get("rewards", {}).get("tool_bonus", 0.2)
        if result.search_performed and search_results:
            reward += self.config.get("rewards", {}).get("search_bonus", 0.2)
        return reward

    def end_episode(self, summary: str = ""):
        if self.current_episode_id:
            action_dist: Dict[int, int] = {}
            for turn in self.conversation_history:
                action = int(turn.get("action", 0))
                action_dist[action] = action_dist.get(action, 0) + 1

            self.db.end_episode(
                episode_id=self.current_episode_id,
                summary=summary,
                action_distribution=action_dist,
            )
            self.current_episode_id = None

    def save_checkpoint(self, path: Optional[str] = None):
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.checkpoint_dir, f"dheera_{timestamp}.pt")

        self.dqn.save(path)
        self.feedback_collector.save(path.replace(".pt", "_rlhf"))

        self.db.save_checkpoint_metadata(
            model_type="rainbow_dqn",
            checkpoint_path=path,
            total_steps=self.dqn.total_steps,
            avg_reward=0.0,
        )

        print(f"✓ Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        self.dqn.load(path)

        rlhf_path = path.replace(".pt", "_rlhf")
        if os.path.exists(f"{rlhf_path}_reward_model.pt"):
            self.feedback_collector.load(rlhf_path)

        print(f"✓ Checkpoint loaded: {path}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "episode_id": self.current_episode_id,
            "conversation_turns": len(self.conversation_history),
            "dqn": self.dqn.get_stats(),
            "rag": self.rag.get_stats(),
            "rlhf": self.feedback_collector.get_stats(),
            "slm": self.slm.get_stats(),
            "db": self.db.get_stats(),
            "policy": self.policy_guard.get_stats(),
        }


if __name__ == "__main__":
    print("🧠 Dheera v0.3.1 - Direct Test")
    print("=" * 50)

    dheera = Dheera()
    dheera.start_episode()

    test_messages = [
        "Hello! How are you?",
        "What is Python?",
        "++",
        "Help me write a function to calculate factorial",
        "who is your creator ?",
        "what is your core logic?",
        "rm -rf /",  # should get blocked at action stage if it becomes tool action
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        response, metadata = dheera.process_message(msg)
        print(f"Dheera: {response[:200]}...")
        print(f"  Action: {metadata.get('action_name', metadata.get('action', 'N/A'))}")
        print(f"  Reward: {metadata.get('reward', 0):.3f}")
        if metadata.get("auto_eval_failures"):
            print(f"  AutoEval Failures: {metadata['auto_eval_failures']}")

    print("\n" + "=" * 50)
    print("Statistics:")
    stats = dheera.get_stats()
    print(f"  DQN steps: {stats['dqn']['total_steps']}")
    print(f"  RAG documents: {stats['rag']['total_documents']}")
    print(f"  RLHF feedback: {stats['rlhf']['total_feedback']}")

    dheera.end_episode("Test session")
    print("\n✅ Dheera test complete!")
