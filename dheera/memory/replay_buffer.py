"""
Dheera Replay Buffer
Experience replay for DQN training with optional prioritization.
"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# Standard transition tuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
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
    
    def sample(self, batch_size: int) -> Transition:
        """Sample a batch of transitions uniformly."""
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    
    def sample_with_indices(self, batch_size: int) -> Tuple[Transition, List[int]]:
        """Sample with indices (for prioritized updates)."""
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        return Transition(*zip(*batch)), indices
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all transitions."""
        self.buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {"size": 0, "capacity": self.capacity}
        
        rewards = [t.reward for t in self.buffer]
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "fill_ratio": len(self.buffer) / self.capacity,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
        }


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
        beta_frames: int = 100_000,  # Frames to anneal beta to 1.0
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        
        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Small constant to ensure non-zero priorities
        self.epsilon = 1e-6
        
    @property
    def beta(self) -> float:
        """Anneal beta from beta_start to 1.0."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
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
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        self.buffer[self.position] = Transition(state, action, reward, next_state, done)
        self.priorities[self.position] = (priority + self.epsilon) ** self.alpha
        
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
        transitions = Transition(*zip(*batch))
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return transitions, weights.astype(np.float32), indices.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


@dataclass
class TransitionWithMetadata:
    """Extended transition with metadata for logging/analysis."""
    state: np.ndarray
    action: int
    action_name: str
    reward_extrinsic: float
    reward_intrinsic: float
    reward_total: float
    next_state: np.ndarray
    done: bool
    
    # Metadata
    user_message: str
    response_snippet: str
    timestamp: float
    episode_id: int
    turn_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "reward_ext": self.reward_extrinsic,
            "reward_int": self.reward_intrinsic,
            "reward_total": self.reward_total,
            "done": self.done,
            "user_message": self.user_message[:100],
            "response_snippet": self.response_snippet[:100],
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "turn_id": self.turn_id,
        }


class LoggingReplayBuffer(ReplayBuffer):
    """
    Replay buffer with extended logging for analysis.
    Stores additional metadata about each transition.
    """
    
    def __init__(self, capacity: int = 100_000):
        super().__init__(capacity)
        self.metadata_buffer: deque = deque(maxlen=capacity)
        
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
    
    def get_recent_metadata(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get metadata for recent transitions."""
        return list(self.metadata_buffer)[-n:]
    
    def export_to_jsonl(self, filepath: str):
        """Export buffer to JSONL for analysis."""
        import json
        
        with open(filepath, 'w') as f:
            for i, (trans, meta) in enumerate(zip(self.buffer, self.metadata_buffer)):
                record = {
                    "index": i,
                    "state": trans.state.tolist() if hasattr(trans.state, 'tolist') else trans.state,
                    "action": trans.action,
                    "reward": trans.reward,
                    "done": trans.done,
                    **meta
                }
                f.write(json.dumps(record) + "\n")


# Quick test
if __name__ == "__main__":
    # Test standard buffer
    buffer = ReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(16).astype(np.float32)
        next_state = np.random.randn(16).astype(np.float32)
        buffer.push(state, random.randint(0, 5), random.random(), next_state, 0.0)
    
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Test sampling
    batch = buffer.sample(32)
    print(f"Sampled batch of {len(batch.state)} transitions")
    
    # Test prioritized buffer
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(16).astype(np.float32)
        next_state = np.random.randn(16).astype(np.float32)
        pri_buffer.push(state, random.randint(0, 5), random.random(), next_state, 0.0)
    
    transitions, weights, indices = pri_buffer.sample(32)
    print(f"Prioritized sample: {len(indices)} transitions, weights range: [{weights.min():.3f}, {weights.max():.3f}]")
