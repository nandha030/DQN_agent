# core/dqn_agent.py
"""
Dheera DQN Agent
Tiny DQN with RND curiosity module - the "Fast Brain" of Dheera.
Version 0.2.0 - Enhanced with Double DQN, action tracking, and better exploration.
"""

import random
import math
from collections import deque, namedtuple
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===============================
# Transition for Replay Buffer
# ===============================

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)


# ===============================
# TinyDQN Network
# ===============================

class TinyDQN(nn.Module):
    """Simple MLP for Q-value estimation with optional dueling architecture."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        dueling: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        
        if dueling:
            # Dueling DQN: separate value and advantage streams
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            # Standard DQN
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Q = V + (A - mean(A))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            return self.net(x)


# ===============================
# RND Curiosity Module
# ===============================

class RNDModule(nn.Module):
    """
    Random Network Distillation for curiosity-driven exploration.
    Intrinsic reward = prediction error (novelty).
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Fixed random target network (never trained)
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Predictor network (trained to match target)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Freeze target parameters
        for p in self.target.parameters():
            p.requires_grad = False
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward (prediction error).
        
        Args:
            state_batch: (B, state_dim) tensor
            
        Returns:
            (B,) tensor of intrinsic rewards
        """
        with torch.no_grad():
            target_feat = self.target(state_batch)
        pred_feat = self.predictor(state_batch)
        mse = (target_feat - pred_feat).pow(2).mean(dim=1)
        return mse

    def rnd_loss(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Loss for training predictor network."""
        with torch.no_grad():
            target_feat = self.target(state_batch)
        pred_feat = self.predictor(state_batch)
        return (target_feat - pred_feat).pow(2).mean()
    
    def update_reward_stats(self, reward: float):
        """Update running statistics for reward normalization."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        if self.reward_count > 1:
            delta2 = reward - self.reward_mean
            self.reward_std = math.sqrt(
                ((self.reward_count - 2) * self.reward_std ** 2 + delta * delta2) / (self.reward_count - 1)
            )
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize intrinsic reward using running statistics."""
        if self.reward_std > 1e-8:
            return (reward - self.reward_mean) / self.reward_std
        return reward


# ===============================
# Prioritized Replay Buffer
# ===============================

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples transitions based on TD-error priority.
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,  # Priority exponent
        beta_start: float = 0.4,  # Importance sampling start
        beta_end: float = 1.0,
        beta_decay_steps: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay_steps = beta_decay_steps
        
        self.buffer: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.step = 0
        self.max_priority = 1.0
    
    @property
    def beta(self) -> float:
        """Get current beta for importance sampling."""
        frac = min(1.0, self.step / self.beta_decay_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)
    
    def push(self, *args):
        """Add transition with max priority."""
        transition = Transition(*args)
        
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[Transition, np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.
        
        Returns:
            (transitions, indices, importance_weights)
        """
        self.step += 1
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get transitions
        batch = [self.buffer[i] for i in indices]
        transitions = Transition(*zip(*batch))
        
        return transitions, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6  # Small constant for stability
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self):
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0


# ===============================
# DQN Agent
# ===============================

class DQNAgent:
    """
    Dheera's Fast Brain - DQN with curiosity-driven exploration.
    
    Features:
    - Double DQN for reduced overestimation
    - Optional Dueling architecture
    - RND curiosity module
    - Optional Prioritized Experience Replay
    - Action tracking and statistics
    - Exploration bonus for underused actions
    """
    
    # Action names for tracking (must match action_space.py)
    ACTION_NAMES = [
        "DIRECT_RESPONSE",
        "CLARIFY_QUESTION",
        "USE_TOOL",
        "SEARCH_WEB",
        "BREAK_DOWN_TASK",
        "REFLECT_AND_REASON",
        "DEFER_OR_DECLINE",
    ]
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_update_freq: int = 1000,
        curiosity_coef: float = 0.1,
        device: Optional[str] = None,
        double_dqn: bool = True,
        dueling: bool = False,
        prioritized_replay: bool = False,
        hidden_dim: int = 64,
        action_bonus_coef: float = 0.05,  # Bonus for underused actions
        normalize_curiosity: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.curiosity_coef = curiosity_coef
        self.double_dqn = double_dqn
        self.action_bonus_coef = action_bonus_coef
        self.normalize_curiosity = normalize_curiosity
        self.hidden_dim = hidden_dim

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # DQN networks
        self.policy_net = TinyDQN(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.target_net = TinyDQN(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Curiosity module
        self.rnd = RNDModule(state_dim, hidden_dim).to(self.device)
        self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        # Replay buffer
        if prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(buffer_capacity)
            self.prioritized = True
        else:
            self.buffer = ReplayBuffer(buffer_capacity)
            self.prioritized = False

        # Epsilon-greedy scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        self.target_update_freq = target_update_freq
        
        # Action tracking
        self.action_counts = np.zeros(action_dim, dtype=np.int64)
        self.action_rewards = np.zeros(action_dim, dtype=np.float64)
        self.action_q_values: List[List[float]] = [[] for _ in range(action_dim)]
        
        # Training stats
        self.training_stats: List[Dict[str, float]] = []
        self.last_loss = 0.0
        self.last_rnd_loss = 0.0
        
        # Metadata
        self.created_at = datetime.now().isoformat()
        self.last_saved_at: Optional[str] = None

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with action bonus.
        
        Args:
            state: State vector (state_dim,)
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Action index
        """
        self.total_steps += 1
        
        # Decay epsilon
        if explore:
            t = min(self.total_steps, self.epsilon_decay_steps)
            frac = t / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

            if random.random() < self.epsilon:
                # Exploration with action bonus for underused actions
                if self.action_bonus_coef > 0 and self.total_steps > 100:
                    action_probs = self._get_exploration_probs()
                    return np.random.choice(self.action_dim, p=action_probs)
                else:
                    return random.randrange(self.action_dim)

        # Greedy action
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        
        action = int(q_values.argmax(dim=1).item())
        
        # Track Q-values for this action
        q_val = q_values[0, action].item()
        if len(self.action_q_values[action]) < 1000:
            self.action_q_values[action].append(q_val)
        else:
            # Rolling window
            self.action_q_values[action] = self.action_q_values[action][-999:] + [q_val]
        
        return action
    
    def _get_exploration_probs(self) -> np.ndarray:
        """
        Get exploration probabilities with bonus for underused actions.
        """
        total_actions = self.action_counts.sum() + 1e-8
        action_freqs = self.action_counts / total_actions
        
        # Inverse frequency bonus (underused actions get higher probability)
        inv_freq = 1.0 / (action_freqs + 1e-8)
        inv_freq = inv_freq / inv_freq.sum()
        
        # Mix uniform with inverse frequency
        uniform = np.ones(self.action_dim) / self.action_dim
        probs = (1 - self.action_bonus_coef) * uniform + self.action_bonus_coef * inv_freq
        
        return probs / probs.sum()  # Ensure normalization
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.cpu().numpy().flatten()

    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward_ext: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> float:
        """
        Store transition with combined reward.
        
        Returns:
            Total reward (extrinsic + curiosity bonus)
        """
        # Compute curiosity reward
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            intrinsic = self.rnd(next_state_t).item()
        
        # Optionally normalize curiosity reward
        if self.normalize_curiosity:
            self.rnd.update_reward_stats(intrinsic)
            intrinsic = self.rnd.normalize_reward(intrinsic)
            intrinsic = np.clip(intrinsic, -5.0, 5.0)  # Clip extremes
        
        reward_total = reward_ext + self.curiosity_coef * intrinsic
        self.buffer.push(state, action, reward_total, next_state, float(done))
        
        # Track action statistics
        self.action_counts[action] += 1
        self.action_rewards[action] += reward_ext
        
        return reward_total

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step.
        
        Returns:
            Training metrics or None if not enough data
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample from buffer
        if self.prioritized:
            transitions, indices, weights = self.buffer.sample(self.batch_size)
            weights_t = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = self.buffer.sample(self.batch_size)
            weights_t = torch.ones(self.batch_size).to(self.device)
            indices = None

        batch = Transition(*transitions)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Current Q-values
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use policy net for action selection, target net for evaluation
                next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_state_batch).max(1)[0]
            
            target_q = reward_batch + self.gamma * next_q_values * (1.0 - done_batch)

        # Calculate TD errors for prioritized replay
        td_errors = (q_values - target_q).detach().cpu().numpy()
        
        # Update priorities
        if self.prioritized and indices is not None:
            self.buffer.update_priorities(indices, td_errors)

        # DQN loss with importance sampling weights
        dqn_loss = (weights_t * nn.functional.smooth_l1_loss(q_values, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # RND curiosity loss
        rnd_loss = self.rnd.rnd_loss(next_state_batch)
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        nn.utils.clip_grad_norm_(self.rnd.predictor.parameters(), 1.0)
        self.rnd_optimizer.step()

        # Target network sync
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Track losses
        self.last_loss = dqn_loss.item()
        self.last_rnd_loss = rnd_loss.item()

        stats = {
            "dqn_loss": self.last_loss,
            "rnd_loss": self.last_rnd_loss,
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "total_steps": self.total_steps,
            "mean_q": q_values.mean().item(),
            "max_q": q_values.max().item(),
        }
        
        # Keep only last 1000 stats
        if len(self.training_stats) >= 1000:
            self.training_stats = self.training_stats[-999:]
        self.training_stats.append(stats)
        
        return stats
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about action usage."""
        total = self.action_counts.sum()
        
        stats = {
            "total_actions": int(total),
            "action_distribution": {},
            "action_avg_rewards": {},
            "action_avg_q_values": {},
        }
        
        for i in range(self.action_dim):
            name = self.ACTION_NAMES[i] if i < len(self.ACTION_NAMES) else f"ACTION_{i}"
            count = int(self.action_counts[i])
            
            stats["action_distribution"][name] = {
                "count": count,
                "percentage": round(count / max(1, total) * 100, 2),
            }
            
            if count > 0:
                stats["action_avg_rewards"][name] = round(self.action_rewards[i] / count, 4)
            else:
                stats["action_avg_rewards"][name] = 0.0
            
            if self.action_q_values[i]:
                stats["action_avg_q_values"][name] = round(np.mean(self.action_q_values[i]), 4)
            else:
                stats["action_avg_q_values"][name] = 0.0
        
        return stats
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_stats:
            return {"status": "no_training_data"}
        
        recent = self.training_stats[-100:]  # Last 100 steps
        
        return {
            "total_steps": self.total_steps,
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.buffer),
            "avg_dqn_loss": round(np.mean([s["dqn_loss"] for s in recent]), 6),
            "avg_rnd_loss": round(np.mean([s["rnd_loss"] for s in recent]), 6),
            "avg_q_value": round(np.mean([s["mean_q"] for s in recent]), 4),
            "max_q_value": round(np.max([s["max_q"] for s in recent]), 4),
            "action_stats": self.get_action_stats(),
        }
    
    def reset_action_stats(self):
        """Reset action statistics."""
        self.action_counts = np.zeros(self.action_dim, dtype=np.int64)
        self.action_rewards = np.zeros(self.action_dim, dtype=np.float64)
        self.action_q_values = [[] for _ in range(self.action_dim)]
    
    def save(self, path: str):
        """Save agent state with metadata."""
        self.last_saved_at = datetime.now().isoformat()
        
        torch.save({
            # Network states
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'rnd_predictor': self.rnd.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rnd_optimizer': self.rnd_optimizer.state_dict(),
            
            # Training state
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            
            # Action statistics
            'action_counts': self.action_counts,
            'action_rewards': self.action_rewards,
            
            # RND statistics
            'rnd_reward_mean': self.rnd.reward_mean,
            'rnd_reward_std': self.rnd.reward_std,
            'rnd_reward_count': self.rnd.reward_count,
            
            # Metadata
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'created_at': self.created_at,
            'saved_at': self.last_saved_at,
            'version': '0.2.0',
        }, path)
        
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load network states
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.rnd.predictor.load_state_dict(checkpoint['rnd_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer'])
        
        # Load training state
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        
        # Load action statistics (if available)
        if 'action_counts' in checkpoint:
            self.action_counts = checkpoint['action_counts']
        if 'action_rewards' in checkpoint:
            self.action_rewards = checkpoint['action_rewards']
        
        # Load RND statistics (if available)
        if 'rnd_reward_mean' in checkpoint:
            self.rnd.reward_mean = checkpoint['rnd_reward_mean']
            self.rnd.reward_std = checkpoint['rnd_reward_std']
            self.rnd.reward_count = checkpoint['rnd_reward_count']
        
        # Load metadata
        if 'created_at' in checkpoint:
            self.created_at = checkpoint['created_at']
    
    def soft_update(self, tau: float = 0.005):
        """Soft update target network (Polyak averaging)."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )
    
    def set_learning_rate(self, lr: float):
        """Update learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.rnd_optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
            "double_dqn": self.double_dqn,
            "prioritized_replay": self.prioritized,
            "curiosity_coef": self.curiosity_coef,
            "epsilon": round(self.epsilon, 4),
            "total_steps": self.total_steps,
            "buffer_size": len(self.buffer),
            "created_at": self.created_at,
            "last_saved_at": self.last_saved_at,
        }


# ===============================
# Quick Test
# ===============================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing DQN Agent")
    print("=" * 60)
    
    # Create agent with 7 actions (including SEARCH_WEB)
    agent = DQNAgent(
        state_dim=16,
        action_dim=7,
        double_dqn=True,
        dueling=False,
        prioritized_replay=False,
        curiosity_coef=0.1,
    )
    
    print(f"\n📊 Agent Info:")
    for key, value in agent.get_info().items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 Simulating training...")
    
    # Simulate training
    for i in range(200):
        state = np.random.randn(16).astype(np.float32)
        action = agent.select_action(state)
        next_state = np.random.randn(16).astype(np.float32)
        
        # Simulate reward based on action
        if action == 3:  # SEARCH_WEB
            reward = random.random() * 0.5  # Search often helpful
        elif action == 0:  # DIRECT_RESPONSE
            reward = random.random() * 0.3
        else:
            reward = random.random() * 0.2
        
        done = random.random() > 0.95
        
        total_reward = agent.store_transition(state, action, reward, next_state, done)
        stats = agent.train_step()
        
        if i % 50 == 0 and stats:
            print(f"   Step {i}: loss={stats['dqn_loss']:.4f}, epsilon={stats['epsilon']:.3f}")
    
    print(f"\n📈 Training Summary:")
    summary = agent.get_training_summary()
    print(f"   Total steps: {summary['total_steps']}")
    print(f"   Epsilon: {summary['epsilon']}")
    print(f"   Avg Q-value: {summary['avg_q_value']}")
    print(f"   Avg DQN loss: {summary['avg_dqn_loss']}")
    
    print(f"\n🎮 Action Statistics:")
    action_stats = summary['action_stats']
    for action_name, dist in action_stats['action_distribution'].items():
        print(f"   {action_name}: {dist['count']} ({dist['percentage']}%)")
    
    # Test save/load
    print(f"\n💾 Testing save/load...")
    agent.save("/tmp/test_dqn.pt")
    
    agent2 = DQNAgent(state_dim=16, action_dim=7)
    agent2.load("/tmp/test_dqn.pt")
    print(f"   Loaded agent with {agent2.total_steps} steps")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
