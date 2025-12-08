"""
Dheera - Main Orchestrator
The central brain that coordinates all components.
"""

import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
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
from connectors.tool_registry import ToolRegistry


@dataclass
class DheeraConfig:
    """Configuration for Dheera."""
    # Agent settings
    name: str = "Dheera"
    version: str = "0.1.0"
    
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
    
    # SLM settings
    slm_provider: str = "ollama"
    slm_model: str = "phi3:mini"
    slm_base_url: str = "http://localhost:11434"
    slm_temperature: float = 0.7
    slm_max_tokens: int = 1024
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    memory_dir: str = "./memory"
    log_dir: str = "./logs"
    
    # Behavior
    auto_save: bool = True
    save_interval: int = 100
    debug_mode: bool = False
    
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
        
        if 'dopamine' in data:
            config.positive_feedback_reward = data['dopamine'].get('positive_feedback_reward', 1.0)
            config.negative_feedback_reward = data['dopamine'].get('negative_feedback_reward', -0.5)
        
        if 'slm' in data:
            config.slm_provider = data['slm'].get('provider', 'ollama')
            config.slm_model = data['slm'].get('model', 'phi3:mini')
            config.slm_base_url = data['slm'].get('base_url', 'http://localhost:11434')
            config.slm_temperature = data['slm'].get('temperature', 0.7)
            config.slm_max_tokens = data['slm'].get('max_tokens', 1024)
        
        return config


class Dheera:
    """
    Dheera - Adaptive AI Agent with Learning Core.
    """
    
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
        self._init_executor()
        self._init_memory()
        self._init_tools()
        
        self.context = ConversationContext()
        self.interaction_count = 0
        self.current_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_action_info: Dict[str, Any] = {}
        
        self.on_action_selected: Optional[Callable] = None
        self.on_response_generated: Optional[Callable] = None
        self.on_feedback_received: Optional[Callable] = None
        
        print(f"🧠 {self.config.name} initialized (v{self.config.version})")
    
    def _setup_directories(self):
        for dir_path in [self.config.checkpoint_dir, self.config.memory_dir, self.config.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _init_action_space(self):
        self.action_space = ActionSpace()
        print(f"  ✓ Action space: {self.action_space.size} actions")
    
    def _init_state_builder(self):
        self.state_builder = StateBuilder(state_dim=self.config.state_dim)
        print(f"  ✓ State builder: {self.config.state_dim} dimensions")
    
    def _init_dqn_agent(self):
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
            print(f"  ✓ DQN loaded from checkpoint (steps: {self.dqn.total_steps})")
        else:
            print(f"  ✓ DQN initialized (fresh)")
    
    def _init_slm(self):
        self.slm = SLMInterface({
            "provider": self.config.slm_provider,
            "model": self.config.slm_model,
            "base_url": self.config.slm_base_url,
            "temperature": self.config.slm_temperature,
            "max_tokens": self.config.slm_max_tokens,
        })
        
        if self.slm.is_available():
            print(f"  ✓ SLM connected: {self.config.slm_provider}/{self.config.slm_model}")
        else:
            print(f"  ⚠ SLM not available: {self.config.slm_provider}/{self.config.slm_model}")
    
    def _init_policy(self):
        self.policy = PolicyGuard()
        print(f"  ✓ Policy guard active")
    
    def _init_executor(self):
        self.executor = ActionExecutor(
            slm=self.slm,
            action_space=self.action_space,
            policy_guard=self.policy,
        )
        print(f"  ✓ Action executor ready")
    
    def _init_memory(self):
        self.replay_buffer = LoggingReplayBuffer(capacity=self.config.buffer_capacity)
        self.episodic_memory = EpisodicMemory(
            storage_path=self.config.memory_dir,
            auto_save=self.config.auto_save,
        )
        stats = self.episodic_memory.get_stats()
        print(f"  ✓ Memory initialized: {stats.get('total_episodes', 0)} episodes loaded")
    
    def _init_tools(self):
        self.tools = ToolRegistry()
        print(f"  ✓ Tools registered: {len(self.tools.list_tools())} available")
    
    def process_message(
        self,
        user_message: str,
        session: Optional[ChatSession] = None,
    ) -> str:
        """Process a user message and generate a response."""
        self.interaction_count += 1
        
        self.context.messages.append({
            "role": "user",
            "content": user_message,
        })
        self.context.turn_count += 1
        
        if session:
            self.context.tools_available = list(self.tools.tools.keys())
        
        state = self.state_builder.build_state(self.context)
        state_info = self.state_builder.state_to_dict(state)
        
        prev_state = self.current_state
        self.current_state = state
        
        action_id = self.dqn.select_action(state)
        action = self.action_space.get_by_id(action_id)
        
        self.last_action_info = {
            "action_id": action_id,
            "action_name": action.name if action else "UNKNOWN",
            "epsilon": self.dqn.epsilon,
            "q_values": self.dqn.get_q_values(state).tolist(),
            "state_summary": {k: round(v, 3) for k, v in state_info.items() if v > 0.1},
        }
        
        if self.on_action_selected:
            self.on_action_selected(self.last_action_info)
        
        result = self.executor.execute(
            action_id=action_id,
            context=self.context,
            user_message=user_message,
            state_info=state_info,
        )
        
        response = result.response
        
        self.context.messages.append({
            "role": "assistant",
            "content": response,
        })
        
        if not self.episodic_memory.current_episode:
            self.episodic_memory.start_episode()
        
        self.episodic_memory.add_turn(
            user_message=user_message,
            assistant_response=response,
            action_taken=action_id,
            action_name=action.name if action else "UNKNOWN",
            reward=0.0,
            state_vector=state,
            metadata={
                "tokens_used": result.metadata.get("tokens_used", 0) if result.metadata else 0,
            }
        )
        
        if prev_state is not None and self.last_action is not None:
            self.dqn.store_transition(
                state=prev_state,
                action=self.last_action,
                reward_ext=self.config.neutral_reward,
                next_state=state,
                done=False,
            )
        
        self.last_action = action_id
        
        train_info = self.dqn.train_step()
        if train_info and self.config.debug_mode:
            self.last_action_info["train_info"] = train_info
        
        if self.config.auto_save and self.interaction_count % self.config.save_interval == 0:
            self.save_checkpoint()
        
        if self.on_response_generated:
            self.on_response_generated(response, result)
        
        return response
    
    def provide_feedback(self, reward: float, session: Optional[ChatSession] = None):
        """Provide feedback for the last interaction."""
        if self.episodic_memory.current_episode and self.episodic_memory.current_episode.entries:
            last_entry = self.episodic_memory.current_episode.entries[-1]
            last_entry.reward = reward
        
        self.context.user_feedback_history.append(reward)
        
        if self.current_state is not None and self.last_action is not None:
            self.dqn.store_transition(
                state=self.current_state,
                action=self.last_action,
                reward_ext=reward,
                next_state=self.current_state,
                done=False,
            )
            self.dqn.train_step()
        
        if self.on_feedback_received:
            self.on_feedback_received(reward)
    
    def end_conversation(self, summary: str = ""):
        """End the current conversation episode."""
        if self.episodic_memory.current_episode:
            self.episodic_memory.end_episode(summary=summary)
        
        self.context = ConversationContext()
        self.current_state = None
        self.last_action = None
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "agent": {
                "name": self.config.name,
                "version": self.config.version,
                "interactions": self.interaction_count,
            },
            "dqn": {
                "total_steps": self.dqn.total_steps,
                "epsilon": self.dqn.epsilon,
                "buffer_size": len(self.dqn.buffer),
            },
            "memory": self.episodic_memory.get_stats(),
            "tools": {
                "available": len(self.tools.list_tools()),
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


def create_dheera(
    config_path: Optional[str] = None,
    slm_provider: str = "ollama",
    slm_model: str = "phi3:mini",
    debug: bool = False,
) -> Dheera:
    """Factory function to create a Dheera instance."""
    if config_path and Path(config_path).exists():
        dheera = Dheera(config_path=config_path)
    else:
        config = DheeraConfig(
            slm_provider=slm_provider,
            slm_model=slm_model,
            debug_mode=debug,
        )
        dheera = Dheera(config=config)
    
    return dheera
