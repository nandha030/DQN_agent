"""
Dheera DQN Agent
Tiny DQN with RND curiosity module - the "Fast Brain" of Dheera.
Based on your working implementation, cleaned up for the project.
"""

import random
from collections import deque, namedtuple
from typing import Optional, Dict, Any, List
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
    """Simple MLP for Q-value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# ===============================
# Replay Buffer
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


# ===============================
# DQN Agent
# ===============================

class DQNAgent:
    """
    Dheera's Fast Brain - DQN with curiosity-driven exploration.
    
    Combines:
    - Extrinsic reward (dopamine from environment/user feedback)
    - Intrinsic reward (curiosity via RND)
    """
    
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
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_capacity)
        self.curiosity_coef = curiosity_coef

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # DQN networks
        self.policy_net = TinyDQN(state_dim, action_dim).to(self.device)
        self.target_net = TinyDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Curiosity module
        self.rnd = RNDModule(state_dim).to(self.device)
        self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        # Epsilon-greedy scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        self.target_update_freq = target_update_freq
        
        # Training stats
        self.training_stats: List[Dict[str, float]] = []

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
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
                return random.randrange(self.action_dim)

        # Greedy action
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())
    
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
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Compute curiosity reward
        with torch.no_grad():
            intrinsic = self.rnd(next_state_t).item()

        reward_total = reward_ext + self.curiosity_coef * intrinsic
        self.buffer.push(state, action, reward_total, next_state, float(done))
        
        return reward_total

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step.
        
        Returns:
            Training metrics or None if not enough data
        """
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*transitions)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # DQN loss (using Huber loss for stability)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + self.gamma * next_q_values * (1.0 - done_batch)

        dqn_loss = nn.functional.smooth_l1_loss(q_values, target_q)

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

        stats = {
            "dqn_loss": dqn_loss.item(),
            "rnd_loss": rnd_loss.item(),
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "total_steps": self.total_steps,
        }
        self.training_stats.append(stats)
        
        return stats
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'rnd_predictor': self.rnd.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rnd_optimizer': self.rnd_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }, path)
        
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.rnd.predictor.load_state_dict(checkpoint['rnd_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']


# Quick test
if __name__ == "__main__":
    agent = DQNAgent(state_dim=16, action_dim=6)
    
    # Simulate a few transitions
    for _ in range(100):
        state = np.random.randn(16).astype(np.float32)
        action = agent.select_action(state)
        next_state = np.random.randn(16).astype(np.float32)
        reward = random.random()
        done = random.random() > 0.9
        
        agent.store_transition(state, action, reward, next_state, done)
        stats = agent.train_step()
        
    if stats:
        print(f"Training stats: {stats}")
    print(f"Agent ready with {agent.action_dim} actions")
