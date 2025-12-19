# dheera.py
"""
Dheera - Main Orchestrator
The central brain that coordinates all components.
Version 0.2.0 - With Web Search Integration
"""

import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Core components
from core.dqn_agent import DQNAgent
from core.action_space import ActionSpace
from core.state_builder import StateBuilder, ConversationContext

# Brain components
from brain.slm_interface import SLMInterface, SLMResponse
from brain.executor import ActionExecutor, ExecutionResult
from brain.policy import PolicyGuard

# Memory components
from memory.replay_buffer import LoggingReplayBuffer
from memory.episodic_memory import EpisodicMemory

# Connectors
from connectors.chat_interface import ChatInterface, ChatSession
from connectors.tool_registry import ToolRegistry, ToolCategory


@dataclass
class DheeraConfig:
    """Configuration for Dheera."""
    # Agent settings
    name: str = "Dheera"
    version: str = "0.2.0"
    
    # DQN settings
    state_dim: int = 16
    hidden_dim: int = 64
    gamma: float = 0.99
    lr: float = 0.001
    batch_size: int = 32
    buffer_capacity: int = 50000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    target_update_freq: int = 500
    
    # Curiosity settings
    curiosity_enabled: bool = True
    curiosity_coef: float = 0.1
    
    # Reward settings
    positive_feedback_reward: float = 1.0
    negative_feedback_reward: float = -0.5
    neutral_reward: float = 0.0
    tool_success_bonus: float = 0.3
    search_relevance_bonus: float = 0.2
    
    # SLM settings
    slm_provider: str = "ollama"
    slm_model: str = "phi3:latest"
    slm_base_url: str = "http://localhost:11434"
    slm_temperature: float = 0.7
    slm_max_tokens: int = 1024
    
    # Tool settings
    tools_config: Dict[str, Any] = field(default_factory=lambda: {
        'web_search': {
            'enabled': True,
            'max_results': 5,
            'timeout': 10,
            'cache_ttl': 300
        }
    })
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    memory_dir: str = "./memory"
    log_dir: str = "./logs"
    
    # Behavior
    auto_save: bool = True
    save_interval: int = 100
    debug_mode: bool = False
    auto_search: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "DheeraConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'agent' in data:
            config.name = data['agent'].get('name', config.name)
            config.version = data['agent'].get('version', config.version)
        
        if 'dqn' in data:
            for key, value in data['dqn'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        if 'curiosity' in data:
            config.curiosity_enabled = data['curiosity'].get('enabled', True)
            config.curiosity_coef = data['curiosity'].get('coefficient', 0.1)
        
        if 'rewards' in data:
            config.positive_feedback_reward = data['rewards'].get('positive_feedback_reward', 1.0)
            config.negative_feedback_reward = data['rewards'].get('negative_feedback_reward', -0.5)
            config.neutral_reward = data['rewards'].get('neutral_reward', 0.0)
            config.tool_success_bonus = data['rewards'].get('tool_success_bonus', 0.3)
            config.search_relevance_bonus = data['rewards'].get('search_relevance_bonus', 0.2)
        
        # Legacy support
        if 'dopamine' in data:
            config.positive_feedback_reward = data['dopamine'].get('positive_feedback_reward', 1.0)
            config.negative_feedback_reward = data['dopamine'].get('negative_feedback_reward', -0.5)
        
        if 'slm' in data:
            config.slm_provider = data['slm'].get('provider', 'ollama')
            config.slm_model = data['slm'].get('model', 'phi3:latest')
            config.slm_base_url = data['slm'].get('base_url', 'http://localhost:11434')
            config.slm_temperature = data['slm'].get('temperature', 0.7)
            config.slm_max_tokens = data['slm'].get('max_tokens', 1024)
        
        if 'tools' in data:
            config.tools_config = data['tools']
        
        if 'behavior' in data:
            config.auto_search = data['behavior'].get('auto_search', True)
        
        return config


class Dheera:
    """
    Dheera - Adaptive AI Agent with Learning Core.
    
    à¤§à¥€à¤° (Dheera) means "Courageous, Wise, Patient" in Sanskrit.
    
    Features:
    - DQN-based action selection with curiosity-driven exploration
    - Web search integration for fact-checking and current information
    - Episodic memory for learning from past conversations
    - Policy guard for safety and alignment
    """
    
    # Action IDs (must match action_space.py)
    ACTION_DIRECT_RESPONSE = 0
    ACTION_CLARIFY_QUESTION = 1
    ACTION_USE_TOOL = 2
    ACTION_SEARCH_WEB = 3
    ACTION_BREAK_DOWN_TASK = 4
    ACTION_REFLECT_AND_REASON = 5
    ACTION_DEFER_OR_DECLINE = 6
    
    def __init__(
        self,
        config: Optional[DheeraConfig] = None,
        config_path: Optional[str] = None,
    ):
        if config_path:
            self.config = DheeraConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = DheeraConfig()
        
        self._setup_directories()
        self._init_action_space()
        self._init_state_builder()
        self._init_dqn_agent()
        self._init_slm()
        self._init_policy()
        self._init_tools()  # Initialize tools BEFORE executor
        self._init_executor()
        self._init_memory()
        
        # Conversation state
        self.context = ConversationContext()
        self.interaction_count = 0
        self.current_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_action_info: Dict[str, Any] = {}
        self.last_search_context: Optional[str] = None
        
        # Callbacks
        self.on_action_selected: Optional[Callable] = None
        self.on_response_generated: Optional[Callable] = None
        self.on_feedback_received: Optional[Callable] = None
        self.on_search_performed: Optional[Callable] = None
        
        print(f"ðŸ§  {self.config.name} initialized (v{self.config.version})")
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.checkpoint_dir, self.config.memory_dir, self.config.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _init_action_space(self):
        """Initialize the action space."""
        self.action_space = ActionSpace()
        print(f"  âœ“ Action space: {self.action_space.size} actions")
    
    def _init_state_builder(self):
        """Initialize the state builder."""
        self.state_builder = StateBuilder(state_dim=self.config.state_dim)
        print(f"  âœ“ State builder: {self.config.state_dim} dimensions")
    
    def _init_dqn_agent(self):
        """Initialize the DQN agent."""
        self.dqn = DQNAgent(
            state_dim=self.config.state_dim,
            action_dim=self.action_space.size,
            gamma=self.config.gamma,
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            buffer_capacity=self.config.buffer_capacity,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay_steps=self.config.epsilon_decay_steps,
            target_update_freq=self.config.target_update_freq,
            curiosity_coef=self.config.curiosity_coef if self.config.curiosity_enabled else 0.0,
        )
        
        checkpoint_path = Path(self.config.checkpoint_dir) / "dheera_dqn.pt"
        if checkpoint_path.exists():
            self.dqn.load(str(checkpoint_path))
            print(f"  âœ“ DQN loaded from checkpoint (steps: {self.dqn.total_steps})")
        else:
            print(f"  âœ“ DQN initialized (fresh)")
    
    def _init_slm(self):
        """Initialize the SLM interface."""
        self.slm = SLMInterface({
            "provider": self.config.slm_provider,
            "model": self.config.slm_model,
            "base_url": self.config.slm_base_url,
            "temperature": self.config.slm_temperature,
            "max_tokens": self.config.slm_max_tokens,
        })
        
        if self.slm.is_available():
            print(f"  âœ“ SLM connected: {self.config.slm_provider}/{self.config.slm_model}")
        else:
            print(f"  âš  SLM not available: {self.config.slm_provider}/{self.config.slm_model}")
    
    def _init_policy(self):
        """Initialize the policy guard."""
        self.policy = PolicyGuard()
        print(f"  âœ“ Policy guard active")
    
    def _init_tools(self):
        """Initialize tool registry with web search."""
        self.tools = ToolRegistry(config={'tools': self.config.tools_config})
        
        search_tools = len([t for t in self.tools.list_tools() if t.category == ToolCategory.SEARCH])
        print(f"  âœ“ Tools registered: {len(self.tools.list_tools())} available ({search_tools} search)")
    
    def _init_executor(self):
        """Initialize the action executor."""
        self.executor = ActionExecutor(
            slm=self.slm,
            action_space=self.action_space,
            policy_guard=self.policy,
        )
        print(f"  âœ“ Action executor ready")
    
    def _init_memory(self):
        """Initialize memory systems."""
        self.replay_buffer = LoggingReplayBuffer(capacity=self.config.buffer_capacity)
        self.episodic_memory = EpisodicMemory(
            storage_path=self.config.memory_dir,
            auto_save=self.config.auto_save,
        )
        stats = self.episodic_memory.get_stats()
        print(f"  âœ“ Memory initialized: {stats.get('total_episodes', 0)} episodes loaded")
    
    def _should_search(self, message: str) -> bool:
        """Determine if a message should trigger web search."""
        if not self.config.auto_search:
            return False
        return self.tools.should_search(message)
    
    def _perform_search(self, query: str, search_type: str = 'general') -> Tuple[bool, str, Dict]:
        """
        Perform a web search and return results.
        
        Args:
            query: Search query
            search_type: 'general', 'news', or 'fact_check'
            
        Returns:
            Tuple of (success, formatted_context, raw_result)
        """
        try:
            if search_type == 'news':
                result = self.tools.execute("search_news", {"query": query})
            elif search_type == 'fact_check':
                result = self.tools.execute("fact_check", {"claim": query})
            else:
                result = self.tools.execute("web_search", {"query": query})
            
            if result.success and result.output:
                formatted = result.output.get('formatted_context', '')
                
                if self.on_search_performed:
                    self.on_search_performed(query, result.output)
                
                return True, formatted, result.output
            else:
                return False, "", {"error": result.error}
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"  [Search error: {e}]")
            return False, "", {"error": str(e)}
    
    def _inject_search_context(self, user_message: str, search_context: str) -> str:
        """Inject search results into the message context for SLM."""
        return f"""The user asked: "{user_message}"

Here is relevant information from the web:

{search_context}

Based on the search results above, provide a helpful and accurate response to the user's question. Cite sources when appropriate."""
    
    def _determine_action(self, state: np.ndarray, user_message: str) -> int:
        """
        Determine the best action using DQN with search heuristics.
        
        This method combines DQN's learned policy with rule-based search detection.
        """
        # Get DQN's action selection
        dqn_action = self.dqn.select_action(state)
        
        # Override with search if strongly indicated and DQN hasn't learned it yet
        if self._should_search(user_message):
            q_values = self.dqn.get_q_values(state)
            
            # Check if SEARCH_WEB action exists in action space
            if self.ACTION_SEARCH_WEB < len(q_values):
                search_q = q_values[self.ACTION_SEARCH_WEB]
                max_q = np.max(q_values)
                
                # If DQN hasn't strongly learned to search, guide it
                if search_q < max_q - 0.3 and self.dqn.epsilon > 0.3:
                    if self.config.debug_mode:
                        print(f"  [Search override: DQN chose {dqn_action}, switching to SEARCH_WEB]")
                    return self.ACTION_SEARCH_WEB
        
        return dqn_action
    
    def process_message(
        self,
        user_message: str,
        session: Optional[ChatSession] = None,
        force_search: bool = False,
    ) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: The user's input message
            session: Optional chat session for context
            force_search: Force web search regardless of action selection
            
        Returns:
            The assistant's response
        """
        self.interaction_count += 1
        
        # Update conversation context
        self.context.messages.append({
            "role": "user",
            "content": user_message,
        })
        self.context.turn_count += 1
        
        if session:
            self.context.tools_available = list(self.tools.tools.keys())
        
        # Build state representation
        state = self.state_builder.build_state(self.context)
        state_info = self.state_builder.state_to_dict(state)
        
        # Store previous state for learning
        prev_state = self.current_state
        self.current_state = state
        
        # Select action using DQN with search heuristics
        action_id = self._determine_action(state, user_message)
        action = self.action_space.get_by_id(action_id)
        
        # Store action info for debugging/callbacks
        self.last_action_info = {
            "action_id": action_id,
            "action_name": action.name if action else "UNKNOWN",
            "epsilon": self.dqn.epsilon,
            "q_values": [f"{v:.3f}" for v in self.dqn.get_q_values(state).tolist()],
            "state_summary": {k: v for k, v in state_info.items() if isinstance(v, (int, float)) and v > 0.1},
        }
        
        if self.on_action_selected:
            self.on_action_selected(self.last_action_info)
        
        # Handle web search action
        search_context = None
        search_bonus = 0.0
        
        if action_id == self.ACTION_SEARCH_WEB or force_search:
            success, search_context, search_result = self._perform_search(user_message)
            
            if success:
                self.last_search_context = search_context
                search_bonus = self.config.search_relevance_bonus
                self.last_action_info["search_performed"] = True
                self.last_action_info["search_results"] = search_result.get('result_count', 0)
                
                if self.config.debug_mode:
                    print(f"  [Search: {search_result.get('result_count', 0)} results]")
            else:
                self.last_action_info["search_performed"] = False
                self.last_action_info["search_error"] = search_result.get('error', 'Unknown')
        
        # Execute action and generate response
        if search_context:
            # Inject search context into the user message for SLM
            enhanced_message = self._inject_search_context(user_message, search_context)
            result = self.executor.execute(
                action_id=action_id,
                context=self.context,
                user_message=enhanced_message,
                state_info=state_info,
            )
        else:
            result = self.executor.execute(
                action_id=action_id,
                context=self.context,
                user_message=user_message,
                state_info=state_info,
            )
        
        response = result.response
        
        # Update conversation context
        self.context.messages.append({
            "role": "assistant",
            "content": response,
        })
        
        # Update episodic memory
        if not self.episodic_memory.current_episode:
            self.episodic_memory.start_episode()
        
        self.episodic_memory.add_turn(
            user_message=user_message,
            assistant_response=response,
            action_taken=action_id,
            action_name=action.name if action else "UNKNOWN",
            reward=search_bonus,
            state_vector=state,
            metadata={
                "tokens_used": result.metadata.get("tokens_used", 0) if result.metadata else 0,
                "search_performed": action_id == self.ACTION_SEARCH_WEB or force_search,
                "search_results": self.last_action_info.get("search_results", 0),
            }
        )
        
        # Store transition for DQN learning
        if prev_state is not None and self.last_action is not None:
            base_reward = self.config.neutral_reward + search_bonus
            
            self.dqn.store_transition(
                state=prev_state,
                action=self.last_action,
                reward_ext=base_reward,
                next_state=state,
                done=False,
            )
        
        self.last_action = action_id
        
        # Train DQN
        train_info = self.dqn.train_step()
        if train_info and self.config.debug_mode:
            self.last_action_info["train_info"] = train_info
        
        # Auto-save checkpoint
        if self.config.auto_save and self.interaction_count % self.config.save_interval == 0:
            self.save_checkpoint()
        
        # Callback
        if self.on_response_generated:
            self.on_response_generated(response, result)
        
        return response
    
    def search(self, query: str, search_type: str = 'general') -> Dict[str, Any]:
        """
        Perform a direct web search (bypassing action selection).
        
        Args:
            query: Search query
            search_type: 'general', 'news', or 'fact_check'
            
        Returns:
            Search results dict
        """
        success, context, result = self._perform_search(query, search_type)
        return {
            'success': success,
            'context': context,
            'results': result
        }
    
    def fact_check(self, claim: str) -> Dict[str, Any]:
        """
        Fact-check a claim using web search.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Fact-check results
        """
        return self.search(claim, search_type='fact_check')
    
    def search_news(self, query: str) -> Dict[str, Any]:
        """
        Search for recent news.
        
        Args:
            query: News search query
            
        Returns:
            News search results
        """
        return self.search(query, search_type='news')
    
    def provide_feedback(self, reward: float, session: Optional[ChatSession] = None):
        """
        Provide feedback for the last interaction.
        
        Args:
            reward: Reward value (positive = good, negative = bad)
            session: Optional chat session
        """
        # Update episodic memory
        if self.episodic_memory.current_episode and self.episodic_memory.current_episode.entries:
            last_entry = self.episodic_memory.current_episode.entries[-1]
            last_entry.reward = reward
        
        # Track feedback history
        self.context.user_feedback_history.append(reward)
        
        # Store feedback transition for learning
        if self.current_state is not None and self.last_action is not None:
            self.dqn.store_transition(
                state=self.current_state,
                action=self.last_action,
                reward_ext=reward,
                next_state=self.current_state,
                done=False,
            )
            
            # Immediate learning from feedback
            train_info = self.dqn.train_step()
            
            if self.config.debug_mode and train_info:
                print(f"  [Feedback learning: reward={reward:.2f}, loss={train_info.get('loss', 'N/A')}]")
        
        # Callback
        if self.on_feedback_received:
            self.on_feedback_received(reward)
    
    def end_conversation(self, summary: str = ""):
        """
        End the current conversation episode.
        
        Args:
            summary: Optional summary of the conversation
        """
        if self.episodic_memory.current_episode:
            self.episodic_memory.end_episode(summary=summary)
        
        # Reset state
        self.context = ConversationContext()
        self.current_state = None
        self.last_action = None
        self.last_search_context = None
        
        if self.config.auto_save:
            self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save DQN checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / "dheera_dqn.pt"
        self.dqn.save(str(checkpoint_path))
        
        if self.config.debug_mode:
            print(f"  [Checkpoint saved: {self.dqn.total_steps} steps]")
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load DQN checkpoint."""
        if path is None:
            path = str(Path(self.config.checkpoint_dir) / "dheera_dqn.pt")
        
        if Path(path).exists():
            self.dqn.load(path)
            print(f"  Checkpoint loaded: {self.dqn.total_steps} steps")
    
    def reset_dqn(self):
        """Reset DQN to fresh state (for testing)."""
        self.dqn = DQNAgent(
            state_dim=self.config.state_dim,
            action_dim=self.action_space.size,
            gamma=self.config.gamma,
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            buffer_capacity=self.config.buffer_capacity,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay_steps=self.config.epsilon_decay_steps,
            target_update_freq=self.config.target_update_freq,
            curiosity_coef=self.config.curiosity_coef if self.config.curiosity_enabled else 0.0,
        )
        print(f"  DQN reset to fresh state")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        tool_stats = self.tools.get_stats()
        
        return {
            "agent": {
                "name": self.config.name,
                "version": self.config.version,
                "interactions": self.interaction_count,
            },
            "dqn": {
                "total_steps": self.dqn.total_steps,
                "epsilon": round(self.dqn.epsilon, 4),
                "buffer_size": len(self.dqn.buffer),
            },
            "memory": self.episodic_memory.get_stats(),
            "tools": {
                "total": tool_stats['total_tools'],
                "search_tools": tool_stats['search_tools'],
                "web_searches": tool_stats['web_search_stats']['total_searches'],
                "cache_size": tool_stats['web_search_stats']['cache_size'],
            },
            "slm": {
                "provider": self.config.slm_provider,
                "model": self.config.slm_model,
                "available": self.slm.is_available(),
            },
        }
    
    def register_tool(self, name: str, function: Callable, description: str, **kwargs):
        """Register a new tool."""
        self.tools.register(name=name, function=function, description=description, **kwargs)
    
    def set_debug_mode(self, enabled: bool):
        """Enable/disable debug mode."""
        self.config.debug_mode = enabled
    
    def get_last_action_info(self) -> Dict[str, Any]:
        """Get information about the last action taken."""
        return self.last_action_info
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get web search statistics."""
        return self.tools.get_stats()['web_search_stats']


def create_dheera(
    config_path: Optional[str] = None,
    slm_provider: str = "ollama",
    slm_model: str = "phi3:latest",
    debug: bool = False,
) -> Dheera:
    """
    Factory function to create a Dheera instance.
    
    Args:
        config_path: Path to YAML config file
        slm_provider: SLM provider (default: ollama)
        slm_model: SLM model name (default: phi3:latest)
        debug: Enable debug mode
        
    Returns:
        Configured Dheera instance
    """
    if config_path and Path(config_path).exists():
        dheera = Dheera(config_path=config_path)
    else:
        config = DheeraConfig(
            slm_provider=slm_provider,
            slm_model=slm_model,
            debug_mode=debug,
        )
        dheera = Dheera(config=config)
    
    if debug:
        dheera.set_debug_mode(True)
    
    return dheera


# ==================== Quick Test ====================
if __name__ == "__main__":
    print("=" * 60)
    print("ðŸ§ª Testing Dheera with Web Search")
    print("=" * 60)
    
    # Create Dheera with debug mode
    dheera = create_dheera(debug=True)
    
    # Test direct search first (doesn't require SLM)
    print("\n" + "-" * 60)
    print("Testing direct search (no SLM required):")
    print("-" * 60)
    
    result = dheera.search("Python programming language")
    print(f"\nSearch success: {result['success']}")
    if result['success']:
        print(f"Results found: {result['results'].get('result_count', 0)}")
        for r in result['results'].get('results', [])[:2]:
            print(f"  - {r['title'][:50]}...")
    else:
        print(f"Error: {result['results'].get('error', 'Unknown')}")
    
    # Test news search
    print("\nNews search:")
    news_result = dheera.search_news("artificial intelligence")
    print(f"News success: {news_result['success']}")
    if news_result['success']:
        print(f"Articles found: {news_result['results'].get('result_count', 0)}")
    
    # Test fact check
    print("\nFact check:")
    fact_result = dheera.fact_check("Python was created by Guido van Rossum")
    print(f"Fact check success: {fact_result['success']}")
    if fact_result['success']:
        print(f"Evidence found: {fact_result['results'].get('evidence_count', 0)}")
    
    # Test conversations (requires SLM)
    print("\n" + "-" * 60)
    print("Testing conversations (requires Ollama):")
    print("-" * 60)
    
    test_messages = [
        "Hello, how are you?",
        "What is the latest version of Python?",
        "Tell me a joke",
    ]
    
    for msg in test_messages:
        print(f"\nðŸ‘¤ User: {msg}")
        try:
            response = dheera.process_message(msg)
            response_preview = response[:150] + "..." if len(response) > 150 else response
            print(f"ðŸ¤– Dheera: {response_preview}")
            
            info = dheera.get_last_action_info()
            print(f"   [Action: {info.get('action_name', 'N/A')}, "
                  f"Search: {info.get('search_performed', False)}]")
        except Exception as e:
            print(f"âŒ Error: {e}")
    
    # Stats
    print("\n" + "-" * 60)
    print("Statistics:")
    print("-" * 60)
    
    stats = dheera.get_stats()
    print(f"  Interactions: {stats['agent']['interactions']}")
    print(f"  DQN Steps: {stats['dqn']['total_steps']}")
    print(f"  Epsilon: {stats['dqn']['epsilon']}")
    print(f"  Web Searches: {stats['tools']['web_searches']}")
    print(f"  Cache Size: {stats['tools']['cache_size']}")
    
    # Cleanup
    dheera.end_conversation("Test session completed")
    
    print("\n" + "=" * 60)
    print("âœ… Test completed!")
    print("=" * 60)
