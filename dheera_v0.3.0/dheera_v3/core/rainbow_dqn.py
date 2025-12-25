# core/rainbow_dqn.py
"""
Dheera v0.3.0 - Rainbow DQN Agent
Combines all 6 DQN improvements for faster, more stable learning.

Components:
1. Double DQN - Reduces overestimation
2. Dueling Networks - Separate value/advantage streams  
3. Noisy Networks - Learned exploration (no epsilon)
4. Prioritized Experience Replay - Focus on important transitions
5. N-step Returns - Better credit assignment
6. Distributional RL (C51) - Learn value distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from collections import deque

from core.curiosity_rnd import CuriosityModule


# ==================== Noisy Linear Layer ====================

class NoisyLinear(nn.Module):
    """
    Noisy Networks for Exploration.
    Adds learnable noise to weights for exploration without epsilon-greedy.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ==================== Rainbow Network ====================

class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network combining:
    - Dueling architecture (value + advantage streams)
    - Noisy layers (for exploration)
    - Distributional output (C51 atoms)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        
        # Support vector for distributional RL
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, atom_size)
        )
        
        # Feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Choose layer type
        Linear = NoisyLinear if noisy else nn.Linear
        
        # Dueling: Value stream
        self.value_hidden = Linear(hidden_dim, hidden_dim)
        self.value_out = Linear(hidden_dim, atom_size)
        
        # Dueling: Advantage stream
        self.advantage_hidden = Linear(hidden_dim, hidden_dim)
        self.advantage_out = Linear(hidden_dim, action_dim * atom_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns Q-value distribution for each action.
        Shape: (batch_size, action_dim, atom_size)
        """
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.feature(x)
        
        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value_out(value).view(batch_size, 1, self.atom_size)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage).view(batch_size, self.action_dim, self.atom_size)
        
        # Combine: Q = V + (A - mean(A))
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Softmax over atoms to get probability distribution
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get expected Q-values from distribution."""
        q_dist = self.forward(x)
        q_values = (q_dist * self.support).sum(dim=-1)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ==================== Rainbow DQN Agent ====================

class RainbowDQNAgent:
    """
    Complete Rainbow DQN Agent for Dheera.
    
    Features:
    - All 6 Rainbow improvements
    - Integrated curiosity module (RND)
    - SQLite-backed replay buffer
    - Action tracking and statistics
    """
    
    ACTION_NAMES = [
        "DIRECT_RESPONSE",
        "CLARIFY_QUESTION",
        "USE_TOOL",
        "SEARCH_WEB",
        "BREAK_DOWN_TASK",
        "REFLECT_AND_REASON",
        "DEFER_OR_DECLINE",
        "COGNITIVE_PROCESS",  # New action for cognitive layer
    ]
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 8,
        hidden_dim: int = 128,
        
        # Rainbow hyperparameters
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        n_step: int = 3,
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        target_update_freq: int = 1000,
        
        # Curiosity
        curiosity_coef: float = 0.1,
        
        # Database
        db_manager: Optional[Any] = None,
        
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_step = n_step
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.target_update_freq = target_update_freq
        self.curiosity_coef = curiosity_coef
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = RainbowNetwork(
            state_dim, action_dim, hidden_dim, atom_size, v_min, v_max
        ).to(self.device)
        
        self.target_net = RainbowNetwork(
            state_dim, action_dim, hidden_dim, atom_size, v_min, v_max
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Curiosity module
        self.curiosity = CuriosityModule(
            state_dim=state_dim,
            intrinsic_reward_scale=curiosity_coef,
            device=self.device,
        )
        
        # Database manager for replay buffer
        self.db = db_manager
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Tracking
        self.total_steps = 0
        self.training_steps = 0
        self.action_counts = np.zeros(action_dim, dtype=np.int64)
        self.action_rewards = np.zeros(action_dim, dtype=np.float64)
        self.episode_rewards = []
        self.training_losses = deque(maxlen=1000)
        
        self.created_at = datetime.now().isoformat()
    
    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using noisy networks (no epsilon needed).
        
        Args:
            state: State vector (state_dim,)
            evaluate: If True, use mean weights (no noise)
            
        Returns:
            (action_id, info_dict)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                self.policy_net.eval()
            
            q_values = self.policy_net.get_q_values(state_t)
            action = q_values.argmax(dim=1).item()
            
            if evaluate:
                self.policy_net.train()
        
        # Get curiosity/novelty score
        novelty = self.curiosity.get_novelty_score(state)
        
        # Track
        self.action_counts[action] += 1
        self.total_steps += 1
        
        info = {
            "q_values": q_values.cpu().numpy().flatten(),
            "q_value": q_values[0, action].item(),
            "novelty": novelty,
            "action_name": self.ACTION_NAMES[action] if action < len(self.ACTION_NAMES) else f"ACTION_{action}",
        }
        
        return action, info
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        episode_id: Optional[str] = None,
    ) -> float:
        """
        Store transition with N-step returns and curiosity.
        
        Returns:
            Total reward (extrinsic + intrinsic)
        """
        # Compute intrinsic reward
        intrinsic_reward = self.curiosity.compute_intrinsic_reward(next_state)
        total_reward = reward + intrinsic_reward
        
        # Track action rewards
        self.action_rewards[action] += total_reward
        
        # N-step transition
        transition = (state, action, total_reward, next_state, done)
        self.n_step_buffer.append(transition)
        
        # Only store when buffer is full or episode ends
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute N-step return
            n_step_reward = 0.0
            for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
            
            # Get first state and last next_state
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]
            
            # Compute priority (use simple TD error estimate)
            with torch.no_grad():
                state_t = torch.FloatTensor(first_state).unsqueeze(0).to(self.device)
                next_state_t = torch.FloatTensor(last_next_state).unsqueeze(0).to(self.device)
                
                q_current = self.policy_net.get_q_values(state_t)[0, first_action].item()
                q_next = self.target_net.get_q_values(next_state_t).max().item()
                
                td_error = abs(n_step_reward + (self.gamma ** self.n_step) * q_next * (1 - last_done) - q_current)
                priority = (td_error + 1e-6) ** 0.6  # alpha = 0.6
            
            # Store in database
            if self.db:
                self.db.store_experience(
                    state=first_state,
                    action=first_action,
                    reward=n_step_reward,
                    next_state=last_next_state,
                    done=last_done,
                    priority=priority,
                    intrinsic_reward=intrinsic_reward,
                    novelty_score=self.curiosity.get_novelty_score(first_state),
                    episode_id=episode_id,
                    n_step_reward=n_step_reward,
                    n_step_next_state=last_next_state,
                    n_step=len(self.n_step_buffer),
                )
        
        # Clear buffer on episode end
        if done:
            self.n_step_buffer.clear()
        
        return total_reward
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step.
        
        Returns:
            Training statistics or None if not enough samples
        """
        if self.db is None:
            return None
        
        # Check if enough experiences
        if self.db.get_experience_count() < self.batch_size:
            return None
        
        # Sample from prioritized replay
        experiences, indices, weights = self.db.sample_experiences(
            batch_size=self.batch_size,
            prioritized=True,
            state_dim=self.state_dim,
        )
        
        if not experiences:
            return None
        
        # Convert to tensors
        states = torch.FloatTensor(
            np.array([e['state'] for e in experiences])
        ).to(self.device)
        
        actions = torch.LongTensor(
            [e['action'] for e in experiences]
        ).to(self.device)
        
        rewards = torch.FloatTensor(
            [e['reward'] for e in experiences]
        ).to(self.device)
        
        next_states = torch.FloatTensor(
            np.array([e['next_state'] for e in experiences])
        ).to(self.device)
        
        dones = torch.FloatTensor(
            [float(e['done']) for e in experiences]
        ).to(self.device)
        
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        # Current Q distribution
        current_dist = self.policy_net(states)
        current_dist = current_dist[range(self.batch_size), actions]  # (batch, atoms)
        
        # Double DQN: use policy net to select, target net to evaluate
        with torch.no_grad():
            # Select best actions with policy net
            next_q_values = self.policy_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Get distribution from target net for selected actions
            next_dist = self.target_net(next_states)
            next_dist = next_dist[range(self.batch_size), next_actions]  # (batch, atoms)
            
            # Project distribution (Categorical DQN)
            support = self.policy_net.support
            delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)
            
            # Compute projected support
            t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_step) * support
            t_z = t_z.clamp(self.v_min, self.v_max)
            
            # Compute projection indices
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle edge cases
            l = l.clamp(0, self.atom_size - 1)
            u = u.clamp(0, self.atom_size - 1)
            
            # Distribute probability
            target_dist = torch.zeros_like(next_dist)
            
            offset = torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long().unsqueeze(1).to(self.device)
            
            target_dist.view(-1).index_add_(
                0, (l + offset).view(-1), 
                (next_dist * (u.float() - b)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (u + offset).view(-1), 
                (next_dist * (b - l.float())).view(-1)
            )
        
        # Cross-entropy loss
        log_p = torch.log(current_dist.clamp(min=1e-8))
        elementwise_loss = -(target_dist * log_p).sum(dim=1)
        
        # Weighted loss (importance sampling)
        loss = (elementwise_loss * weights_t).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        new_priorities = elementwise_loss.detach().cpu().numpy() + 1e-6
        self.db.update_priorities(indices, new_priorities.tolist())
        
        # Reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Train curiosity module
        curiosity_stats = self.curiosity.train_step(
            np.array([e['state'] for e in experiences])
        )
        
        # Track
        loss_val = loss.item()
        self.training_losses.append(loss_val)
        
        return {
            "loss": loss_val,
            "mean_q": self.policy_net.get_q_values(states).mean().item(),
            "max_q": self.policy_net.get_q_values(states).max().item(),
            "rnd_loss": curiosity_stats.get("rnd_loss", 0),
            "training_steps": self.training_steps,
        }
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get action usage statistics."""
        total = self.action_counts.sum()
        
        stats = {
            "total_actions": int(total),
            "actions": {},
        }
        
        for i in range(self.action_dim):
            name = self.ACTION_NAMES[i] if i < len(self.ACTION_NAMES) else f"ACTION_{i}"
            count = int(self.action_counts[i])
            
            stats["actions"][name] = {
                "count": count,
                "percentage": round(count / max(1, total) * 100, 2),
                "avg_reward": round(self.action_rewards[i] / max(1, count), 4),
            }
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        curiosity_stats = self.curiosity.get_stats()
        
        return {
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "avg_loss": np.mean(self.training_losses) if self.training_losses else 0,
            "action_stats": self.get_action_stats(),
            "curiosity": curiosity_stats,
            "device": self.device,
            "created_at": self.created_at,
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "action_counts": self.action_counts,
            "action_rewards": self.action_rewards,
            "created_at": self.created_at,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "n_step": self.n_step,
                "atom_size": self.atom_size,
            },
        }, path)
        
        # Save curiosity separately
        curiosity_path = path.replace(".pt", "_curiosity.pt")
        self.curiosity.save(curiosity_path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        self.total_steps = checkpoint["total_steps"]
        self.training_steps = checkpoint["training_steps"]
        self.action_counts = checkpoint["action_counts"]
        self.action_rewards = checkpoint["action_rewards"]
        self.created_at = checkpoint.get("created_at", self.created_at)
        
        # Load curiosity
        curiosity_path = path.replace(".pt", "_curiosity.pt")
        try:
            self.curiosity.load(curiosity_path)
        except FileNotFoundError:
            print("Warning: Curiosity checkpoint not found, using fresh curiosity module")


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing RainbowDQNAgent...")
    
    # Create agent without database
    agent = RainbowDQNAgent(
        state_dim=64,
        action_dim=8,
        db_manager=None,  # No DB for simple test
    )
    
    print(f"✓ Agent created on device: {agent.device}")
    
    # Test action selection
    state = np.random.randn(64).astype(np.float32)
    action, info = agent.select_action(state)
    print(f"✓ Selected action: {action} ({info['action_name']})")
    print(f"  Q-value: {info['q_value']:.4f}")
    print(f"  Novelty: {info['novelty']:.4f}")
    
    # Test multiple actions
    for i in range(100):
        state = np.random.randn(64).astype(np.float32)
        action, _ = agent.select_action(state)
    
    action_stats = agent.get_action_stats()
    print(f"✓ Action distribution after 100 steps:")
    for name, data in list(action_stats["actions"].items())[:3]:
        print(f"  {name}: {data['count']} ({data['percentage']}%)")
    
    # Test stats
    stats = agent.get_stats()
    print(f"✓ Total steps: {stats['total_steps']}")
    print(f"✓ Curiosity stats: {stats['curiosity']}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        agent.save(f.name)
        
        agent2 = RainbowDQNAgent(state_dim=64, action_dim=8)
        agent2.load(f.name)
        print(f"✓ Loaded agent with {agent2.total_steps} steps")
    
    print("\n✅ All Rainbow DQN tests passed!")
