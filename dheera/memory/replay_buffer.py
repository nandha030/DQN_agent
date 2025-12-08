# memory/replay_buffer.py
"""
Dheera Replay Buffer
Experience replay for DQN training with optional prioritization.
Version 0.2.0 - Enhanced with search tracking and 7-action support.
"""

import random
import time
import json
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Action names (must match action_space.py)
ACTION_NAMES = [
    "DIRECT_RESPONSE",
    "CLARIFY_QUESTION",
    "USE_TOOL",
    "SEARCH_WEB",
    "BREAK_DOWN_TASK",
    "REFLECT_AND_REASON",
    "DEFER_OR_DECLINE",
]


# Standard transition tuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)


# Extended transition with metadata
ExtendedTransition = namedtuple(
    'ExtendedTransition', (
        'state', 'action', 'reward', 'next_state', 'done',
        'search_performed', 'tool_used', 'timestamp'
    )
)


class ReplayBuffer:
    """
    Standard experience replay buffer for DQN.
    Stores transitions and samples uniformly for training.
    """
    
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.position = 0
        
        # Statistics tracking
        self._action_counts = np.zeros(7, dtype=np.int64)
        self._reward_sum = 0.0
        self._reward_sq_sum = 0.0
        self._search_count = 0
        self._tool_count = 0
        
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float
    ):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
        
        # Update stats
        if action < len(self._action_counts):
            self._action_counts[action] += 1
        self._reward_sum += reward
        self._reward_sq_sum += reward * reward
    
    def push_extended(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        search_performed: bool = False,
        tool_used: Optional[str] = None,
    ):
        """Add a transition with extended metadata."""
        self.buffer.append(ExtendedTransition(
            state, action, reward, next_state, done,
            search_performed, tool_used, time.time()
        ))
        
        # Update stats
        if action < len(self._action_counts):
            self._action_counts[action] += 1
        self._reward_sum += reward
        self._reward_sq_sum += reward * reward
        
        if search_performed:
            self._search_count += 1
        if tool_used:
            self._tool_count += 1
    
    def sample(self, batch_size: int) -> Transition:
        """Sample a batch of transitions uniformly."""
        batch = random.sample(self.buffer, batch_size)
        
        # Handle both Transition and ExtendedTransition
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]
        
        return Transition(states, actions, rewards, next_states, dones)
    
    def sample_with_indices(self, batch_size: int) -> Tuple[Transition, List[int]]:
        """Sample with indices (for prioritized updates)."""
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]
        
        return Transition(states, actions, rewards, next_states, dones), indices
    
    def sample_by_action(self, action_id: int, batch_size: int) -> Optional[Transition]:
        """Sample transitions for a specific action."""
        matching = [t for t in self.buffer if t.action == action_id]
        
        if len(matching) < batch_size:
            return None
        
        batch = random.sample(matching, batch_size)
        
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]
        
        return Transition(states, actions, rewards, next_states, dones)
    
    def sample_search_transitions(self, batch_size: int) -> Optional[Transition]:
        """Sample transitions where search was performed."""
        matching = [
            t for t in self.buffer 
            if hasattr(t, 'search_performed') and t.search_performed
        ]
        
        if len(matching) < batch_size:
            # Fall back to action-based filtering
            matching = [t for t in self.buffer if t.action == 3]  # SEARCH_WEB
        
        if len(matching) < batch_size:
            return None
        
        batch = random.sample(matching, batch_size)
        
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]
        
        return Transition(states, actions, rewards, next_states, dones)
    
    def sample_high_reward(self, batch_size: int, threshold: float = 0.5) -> Optional[Transition]:
        """Sample transitions with reward above threshold."""
        matching = [t for t in self.buffer if t.reward > threshold]
        
        if len(matching) < batch_size:
            return None
        
        batch = random.sample(matching, batch_size)
        
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]
        
        return Transition(states, actions, rewards, next_states, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all transitions."""
        self.buffer.clear()
        self._action_counts = np.zeros(7, dtype=np.int64)
        self._reward_sum = 0.0
        self._reward_sq_sum = 0.0
        self._search_count = 0
        self._tool_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        size = len(self.buffer)
        
        if size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "fill_ratio": 0.0,
            }
        
        rewards = [t.reward for t in self.buffer]
        
        # Action distribution
        action_dist = {}
        total_actions = self._action_counts.sum()
        for i, count in enumerate(self._action_counts):
            if count > 0:
                name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"ACTION_{i}"
                action_dist[name] = {
                    "count": int(count),
                    "percentage": round(count / max(1, total_actions) * 100, 1),
                }
        
        return {
            "size": size,
            "capacity": self.capacity,
            "fill_ratio": round(size / self.capacity, 4),
            "reward_mean": round(float(np.mean(rewards)), 4),
            "reward_std": round(float(np.std(rewards)), 4),
            "reward_min": round(float(np.min(rewards)), 4),
            "reward_max": round(float(np.max(rewards)), 4),
            "search_count": self._search_count,
            "tool_count": self._tool_count,
            "action_distribution": action_dist,
        }
    
    def get_action_rewards(self) -> Dict[int, Dict[str, float]]:
        """Get reward statistics per action."""
        action_rewards: Dict[int, List[float]] = {i: [] for i in range(7)}
        
        for t in self.buffer:
            if t.action < 7:
                action_rewards[t.action].append(t.reward)
        
        stats = {}
        for action_id, rewards in action_rewards.items():
            if rewards:
                name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"ACTION_{action_id}"
                stats[action_id] = {
                    "name": name,
                    "count": len(rewards),
                    "mean": round(float(np.mean(rewards)), 4),
                    "std": round(float(np.std(rewards)), 4),
                    "min": round(float(np.min(rewards)), 4),
                    "max": round(float(np.max(rewards)), 4),
                }
        
        return stats


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples transitions based on TD-error priority.
    
    Higher error = more learning potential = higher sample probability.
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,  # Priority exponent
        beta_start: float = 0.4,  # Importance sampling start
        beta_end: float = 1.0,
        beta_frames: int = 100_000,  # Frames to anneal beta to 1.0
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0
        
        self.buffer: List[Optional[Union[Transition, ExtendedTransition]]] = [None] * capacity
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Small constant to ensure non-zero priorities
        self.epsilon = 1e-6
        
        # Statistics tracking
        self._action_counts = np.zeros(7, dtype=np.int64)
        self._search_count = 0
        self._tool_count = 0
        self._max_priority = 1.0
        
    @property
    def beta(self) -> float:
        """Anneal beta from beta_start to beta_end."""
        frac = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + frac * (self.beta_end - self.beta_start)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        priority: Optional[float] = None
    ):
        """Add a transition with priority."""
        # Default to max priority for new transitions
        if priority is None:
            priority = self._max_priority
        
        self.buffer[self.position] = Transition(state, action, reward, next_state, done)
        self.priorities[self.position] = (priority + self.epsilon) ** self.alpha
        
        # Update max priority
        self._max_priority = max(self._max_priority, priority)
        
        # Update stats
        if action < len(self._action_counts):
            self._action_counts[action] += 1
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame += 1
    
    def push_extended(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        search_performed: bool = False,
        tool_used: Optional[str] = None,
        priority: Optional[float] = None
    ):
        """Add a transition with extended metadata and priority."""
        if priority is None:
            priority = self._max_priority
        
        self.buffer[self.position] = ExtendedTransition(
            state, action, reward, next_state, done,
            search_performed, tool_used, time.time()
        )
        self.priorities[self.position] = (priority + self.epsilon) ** self.alpha
        
        # Update max priority
        self._max_priority = max(self._max_priority, priority)
        
        # Update stats
        if action < len(self._action_counts):
            self._action_counts[action] += 1
        if search_performed:
            self._search_count += 1
        if tool_used:
            self._tool_count += 1
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame += 1
    
    def sample(self, batch_size: int) -> Tuple[Transition, np.ndarray, List[int]]:
        """
        Sample a prioritized batch.
        
        Returns:
            - Transition batch
            - Importance sampling weights
            - Indices for priority updates
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Gather transitions
        batch = [self.buffer[i] for i in indices]
        
        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        rewards = [t.reward for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]
        
        transitions = Transition(states, actions, rewards, next_states, dones)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return transitions, weights.astype(np.float32), indices.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + self.epsilon
            self.priorities[idx] = priority ** self.alpha
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
    
    def clear(self):
        """Clear all transitions."""
        self.buffer = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self._action_counts = np.zeros(7, dtype=np.int64)
        self._search_count = 0
        self._tool_count = 0
        self._max_priority = 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "fill_ratio": 0.0,
            }
        
        valid_transitions = [t for t in self.buffer[:self.size] if t is not None]
        rewards = [t.reward for t in valid_transitions]
        
        # Action distribution
        action_dist = {}
        total_actions = self._action_counts.sum()
        for i, count in enumerate(self._action_counts):
            if count > 0:
                name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"ACTION_{i}"
                action_dist[name] = {
                    "count": int(count),
                    "percentage": round(count / max(1, total_actions) * 100, 1),
                }
        
        return {
            "size": self.size,
            "capacity": self.capacity,
            "fill_ratio": round(self.size / self.capacity, 4),
            "frame": self.frame,
            "beta": round(self.beta, 4),
            "max_priority": round(self._max_priority, 4),
            "reward_mean": round(float(np.mean(rewards)), 4) if rewards else 0.0,
            "reward_std": round(float(np.std(rewards)), 4) if rewards else 0.0,
            "search_count": self._search_count,
            "tool_count": self._tool_count,
            "action_distribution": action_dist,
        }


@dataclass
class TransitionWithMetadata:
    """Extended transition with full metadata for logging/analysis."""
    state: np.ndarray
    action: int
    action_name: str
    reward_extrinsic: float
    reward_intrinsic: float
    reward_total: float
    next_state: np.ndarray
    done: bool
    
    # Metadata
    user_message: str = ""
    response_snippet: str = ""
    timestamp: float = field(default_factory=time.time)
    episode_id: str = ""
    turn_id: int = 0
    
    # Search/tool tracking
    search_performed: bool = False
    search_query: Optional[str] = None
    search_result_count: int = 0
    tool_used: Optional[str] = None
    response_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "reward_ext": self.reward_extrinsic,
            "reward_int": self.reward_intrinsic,
            "reward_total": self.reward_total,
            "done": self.done,
            "user_message": self.user_message[:100] if self.user_message else "",
            "response_snippet": self.response_snippet[:100] if self.response_snippet else "",
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "turn_id": self.turn_id,
            "search_performed": self.search_performed,
            "search_query": self.search_query,
            "search_result_count": self.search_result_count,
            "tool_used": self.tool_used,
            "response_latency_ms": self.response_latency_ms,
        }
    
    @property
    def is_search_action(self) -> bool:
        return self.action == 3 or self.search_performed
    
    @property
    def is_tool_action(self) -> bool:
        return self.action == 2 or self.tool_used is not None


class LoggingReplayBuffer(ReplayBuffer):
    """
    Replay buffer with extended logging for analysis.
    Stores additional metadata about each transition.
    """
    
    def __init__(self, capacity: int = 100_000):
        super().__init__(capacity)
        self.metadata_buffer: deque = deque(maxlen=capacity)
        self.extended_buffer: deque = deque(maxlen=capacity)
        
    def push_with_metadata(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        metadata: Dict[str, Any]
    ):
        """Push transition with associated metadata."""
        super().push(state, action, reward, next_state, done)
        self.metadata_buffer.append(metadata)
        
        # Track search/tool
        if metadata.get("search_performed"):
            self._search_count += 1
        if metadata.get("tool_used"):
            self._tool_count += 1
    
    def push_extended_transition(self, transition: TransitionWithMetadata):
        """Push a fully extended transition."""
        super().push(
            transition.state,
            transition.action,
            transition.reward_total,
            transition.next_state,
            float(transition.done)
        )
        self.extended_buffer.append(transition)
        self.metadata_buffer.append(transition.to_dict())
        
        # Track search/tool
        if transition.search_performed:
            self._search_count += 1
        if transition.tool_used:
            self._tool_count += 1
    
    def get_recent_metadata(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get metadata for recent transitions."""
        return list(self.metadata_buffer)[-n:]
    
    def get_recent_extended(self, n: int = 10) -> List[TransitionWithMetadata]:
        """Get recent extended transitions."""
        return list(self.extended_buffer)[-n:]
    
    def get_search_transitions(self) -> List[TransitionWithMetadata]:
        """Get all transitions where search was performed."""
        return [t for t in self.extended_buffer if t.is_search_action]
    
    def get_tool_transitions(self) -> List[TransitionWithMetadata]:
        """Get all transitions where a tool was used."""
        return [t for t in self.extended_buffer if t.is_tool_action]
    
    def get_high_reward_transitions(self, threshold: float = 0.5) -> List[TransitionWithMetadata]:
        """Get transitions with high reward."""
        return [t for t in self.extended_buffer if t.reward_total > threshold]
    
    def analyze_action_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by action type."""
        action_data: Dict[int, List[TransitionWithMetadata]] = {i: [] for i in range(7)}
        
        for t in self.extended_buffer:
            if t.action < 7:
                action_data[t.action].append(t)
        
        analysis = {}
        for action_id, transitions in action_data.items():
            if not transitions:
                continue
            
            name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"ACTION_{action_id}"
            rewards = [t.reward_total for t in transitions]
            latencies = [t.response_latency_ms for t in transitions if t.response_latency_ms > 0]
            
            analysis[name] = {
                "count": len(transitions),
                "reward_mean": round(float(np.mean(rewards)), 4),
                "reward_std": round(float(np.std(rewards)), 4),
                "success_rate": round(float(np.mean([r > 0.5 for r in rewards])), 4),
                "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            }
            
            # Search-specific stats
            if action_id == 3:
                search_results = [t.search_result_count for t in transitions]
                analysis[name]["avg_search_results"] = round(float(np.mean(search_results)), 2)
        
        return analysis
    
    def export_to_jsonl(self, filepath: str):
        """Export buffer to JSONL for analysis."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            for i, trans in enumerate(self.buffer):
                if trans is None:
                    continue
                
                # Get metadata if available
                meta = self.metadata_buffer[i] if i < len(self.metadata_buffer) else {}
                
                record = {
                    "index": i,
                    "state": trans.state.tolist() if hasattr(trans.state, 'tolist') else list(trans.state),
                    "action": trans.action,
                    "action_name": ACTION_NAMES[trans.action] if trans.action < len(ACTION_NAMES) else f"ACTION_{trans.action}",
                    "reward": trans.reward,
                    "done": trans.done,
                    **meta
                }
                f.write(json.dumps(record) + "\n")
    
    def export_search_analysis(self, filepath: str):
        """Export search-specific analysis."""
        search_data = []
        
        for t in self.extended_buffer:
            if t.is_search_action:
                search_data.append({
                    "timestamp": t.timestamp,
                    "user_message": t.user_message,
                    "search_query": t.search_query,
                    "result_count": t.search_result_count,
                    "reward": t.reward_total,
                    "successful": t.reward_total > 0.5,
                    "latency_ms": t.response_latency_ms,
                })
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                "total_searches": len(search_data),
                "success_rate": np.mean([s["successful"] for s in search_data]) if search_data else 0,
                "searches": search_data,
            }, f, indent=2)


class HindsightReplayBuffer(ReplayBuffer):
    """
    Replay buffer with Hindsight Experience Replay (HER).
    Allows relabeling failed experiences as successful with modified goals.
    """
    
    def __init__(self, capacity: int = 100_000, her_ratio: float = 0.5):
        super().__init__(capacity)
        self.her_ratio = her_ratio
        self.episode_buffer: List[List[Transition]] = []
        self.current_episode: List[Transition] = []
    
    def push_to_episode(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float
    ):
        """Add to current episode."""
        trans = Transition(state, action, reward, next_state, done)
        self.current_episode.append(trans)
        
        if done:
            self.end_episode()
    
    def end_episode(self):
        """End current episode and add to buffer with HER."""
        if not self.current_episode:
            return
        
        # Add original transitions
        for trans in self.current_episode:
            self.push(trans.state, trans.action, trans.reward, trans.next_state, trans.done)
        
        # Add HER transitions (relabeled)
        if self.her_ratio > 0 and len(self.current_episode) > 1:
            num_her = int(len(self.current_episode) * self.her_ratio)
            
            for i in range(min(num_her, len(self.current_episode) - 1)):
                # Use a future state as the "achieved goal"
                future_idx = random.randint(i + 1, len(self.current_episode) - 1)
                future_state = self.current_episode[future_idx].state
                
                # Original transition
                orig = self.current_episode[i]
                
                # Relabel with higher reward (pretend we achieved the goal)
                relabeled_reward = 1.0  # Success reward
                
                self.push(orig.state, orig.action, relabeled_reward, future_state, 0.0)
        
        # Store episode for potential analysis
        self.episode_buffer.append(self.current_episode)
        if len(self.episode_buffer) > 100:  # Keep last 100 episodes
            self.episode_buffer.pop(0)
        
        self.current_episode = []


# ===============================
# Quick Test
# ===============================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Replay Buffers")
    print("=" * 60)
    
    # Test standard buffer
    print("\n📦 Testing Standard Buffer...")
    buffer = ReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(16).astype(np.float32)
        next_state = np.random.randn(16).astype(np.float32)
        action = random.randint(0, 6)  # 7 actions
        buffer.push(state, action, random.random(), next_state, 0.0)
    
    print(f"   Buffer size: {len(buffer)}")
    stats = buffer.get_stats()
    print(f"   Reward mean: {stats['reward_mean']}")
    print(f"   Action distribution: {list(stats['action_distribution'].keys())}")
    
    # Test sampling
    batch = buffer.sample(32)
    print(f"   Sampled batch of {len(batch.state)} transitions")
    
    # Test action rewards
    action_rewards = buffer.get_action_rewards()
    print(f"   Actions tracked: {len(action_rewards)}")
    
    # Test prioritized buffer
    print("\n📊 Testing Prioritized Buffer...")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(16).astype(np.float32)
        next_state = np.random.randn(16).astype(np.float32)
        action = random.randint(0, 6)
        pri_buffer.push(state, action, random.random(), next_state, 0.0)
    
    transitions, weights, indices = pri_buffer.sample(32)
    print(f"   Prioritized sample: {len(indices)} transitions")
    print(f"   Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"   Beta: {pri_buffer.beta:.4f}")
    
    # Update priorities
    td_errors = np.random.randn(32)
    pri_buffer.update_priorities(indices, td_errors)
    print(f"   Max priority after update: {pri_buffer._max_priority:.4f}")
    
    # Test logging buffer
    print("\n📝 Testing Logging Buffer...")
    log_buffer = LoggingReplayBuffer(capacity=1000)
    
    for i in range(50):
        state = np.random.randn(16).astype(np.float32)
        next_state = np.random.randn(16).astype(np.float32)
        action = random.randint(0, 6)
        search_performed = (action == 3)
        
        trans = TransitionWithMetadata(
            state=state,
            action=action,
            action_name=ACTION_NAMES[action],
            reward_extrinsic=random.random(),
            reward_intrinsic=random.random() * 0.1,
            reward_total=random.random(),
            next_state=next_state,
            done=False,
            user_message=f"Test message {i}",
            response_snippet=f"Test response {i}",
            episode_id=f"ep_{i // 10}",
            turn_id=i % 10,
            search_performed=search_performed,
            search_query=f"search query {i}" if search_performed else None,
            search_result_count=5 if search_performed else 0,
        )
        
        log_buffer.push_extended_transition(trans)
    
    print(f"   Logging buffer size: {len(log_buffer)}")
    print(f"   Search transitions: {len(log_buffer.get_search_transitions())}")
    
    analysis = log_buffer.analyze_action_performance()
    print(f"   Actions analyzed: {list(analysis.keys())}")
    
    if "SEARCH_WEB" in analysis:
        print(f"   SEARCH_WEB stats: {analysis['SEARCH_WEB']}")
    
    # Test HER buffer
    print("\n🔄 Testing HER Buffer...")
    her_buffer = HindsightReplayBuffer(capacity=1000, her_ratio=0.5)
    
    for ep in range(5):
        for step in range(10):
            state = np.random.randn(16).astype(np.float32)
            next_state = np.random.randn(16).astype(np.float32)
            done = (step == 9)
            her_buffer.push_to_episode(state, random.randint(0, 6), random.random(), next_state, float(done))
    
    print(f"   HER buffer size: {len(her_buffer)} (includes relabeled)")
    print(f"   Episodes stored: {len(her_buffer.episode_buffer)}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
