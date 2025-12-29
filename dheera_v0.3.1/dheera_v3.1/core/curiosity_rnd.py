# core/curiosity_rnd.py
"""
Dheera v0.3.0 - Random Network Distillation (RND) Curiosity Module
Provides intrinsic motivation for exploration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Tuple
from collections import deque


class RNDNetwork(nn.Module):
    """
    RND Network pair: fixed target + trainable predictor.
    Intrinsic reward = prediction error (novelty).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        # Fixed random target network (never trained)
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Trainable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Initialize target with random weights
        self._init_target()
    
    def _init_target(self):
        """Initialize target network with random weights."""
        for module in self.target.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks.
        
        Returns:
            (target_output, predictor_output)
        """
        with torch.no_grad():
            target_out = self.target(x)
        predictor_out = self.predictor(x)
        return target_out, predictor_out
    
    def get_intrinsic_reward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward as prediction error.
        
        Args:
            x: State tensor (batch_size, input_dim)
            
        Returns:
            Intrinsic rewards (batch_size,)
        """
        target_out, predictor_out = self.forward(x)
        intrinsic_reward = (target_out - predictor_out).pow(2).mean(dim=-1)
        return intrinsic_reward
    
    def get_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction loss for training predictor.
        """
        target_out, predictor_out = self.forward(x)
        loss = (target_out - predictor_out).pow(2).mean()
        return loss


class RunningMeanStd:
    """
    Running mean and standard deviation for reward normalization.
    """
    
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update running stats with new batch."""
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.var = m2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values using running stats."""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class CuriosityModule:
    """
    Complete curiosity system for Dheera.
    
    Features:
    - RND-based intrinsic rewards
    - Reward normalization
    - Novelty tracking
    - Curiosity decay for familiar states
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        lr: float = 1e-4,
        intrinsic_reward_scale: float = 0.1,
        normalize_rewards: bool = True,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.normalize_rewards = normalize_rewards
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # RND networks
        self.rnd = RNDNetwork(state_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)
        
        # Reward normalization
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd()
        
        # Tracking
        self.total_intrinsic_reward = 0.0
        self.intrinsic_reward_history = deque(maxlen=1000)
        self.novelty_history = deque(maxlen=1000)
        self.training_steps = 0
    
    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        update_stats: bool = True,
    ) -> float:
        """
        Compute intrinsic reward for a single state.
        
        Args:
            state: State vector (state_dim,)
            update_stats: Whether to update running stats
            
        Returns:
            Scaled intrinsic reward
        """
        # Normalize observation
        if self.normalize_rewards:
            self.obs_rms.update(state.reshape(1, -1))
            normalized_state = self.obs_rms.normalize(state)
        else:
            normalized_state = state
        
        # Convert to tensor
        state_t = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
        
        # Get intrinsic reward
        with torch.no_grad():
            intrinsic_reward = self.rnd.get_intrinsic_reward(state_t).item()
        
        # Normalize reward
        if self.normalize_rewards and update_stats:
            self.reward_rms.update(np.array([intrinsic_reward]))
            intrinsic_reward = self.reward_rms.normalize(np.array([intrinsic_reward]))[0]
        
        # Scale
        scaled_reward = intrinsic_reward * self.intrinsic_reward_scale
        
        # Track
        if update_stats:
            self.total_intrinsic_reward += scaled_reward
            self.intrinsic_reward_history.append(scaled_reward)
            self.novelty_history.append(intrinsic_reward)
        
        return scaled_reward
    
    def compute_batch_intrinsic_rewards(
        self,
        states: np.ndarray,
    ) -> np.ndarray:
        """
        Compute intrinsic rewards for a batch of states.
        
        Args:
            states: State array (batch_size, state_dim)
            
        Returns:
            Intrinsic rewards (batch_size,)
        """
        # Normalize observations
        if self.normalize_rewards:
            self.obs_rms.update(states)
            normalized_states = self.obs_rms.normalize(states)
        else:
            normalized_states = states
        
        # Convert to tensor
        states_t = torch.FloatTensor(normalized_states).to(self.device)
        
        # Get intrinsic rewards
        with torch.no_grad():
            intrinsic_rewards = self.rnd.get_intrinsic_reward(states_t).cpu().numpy()
        
        # Normalize rewards
        if self.normalize_rewards:
            self.reward_rms.update(intrinsic_rewards)
            intrinsic_rewards = self.reward_rms.normalize(intrinsic_rewards)
        
        # Scale
        return intrinsic_rewards * self.intrinsic_reward_scale
    
    def train_step(self, states: np.ndarray) -> Dict[str, float]:
        """
        Train the predictor network on a batch of states.
        
        Args:
            states: State array (batch_size, state_dim)
            
        Returns:
            Training statistics
        """
        # Normalize observations
        if self.normalize_rewards:
            normalized_states = self.obs_rms.normalize(states)
        else:
            normalized_states = states
        
        # Convert to tensor
        states_t = torch.FloatTensor(normalized_states).to(self.device)
        
        # Compute loss
        loss = self.rnd.get_loss(states_t)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd.predictor.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        return {
            "rnd_loss": loss.item(),
            "training_steps": self.training_steps,
        }
    
    def get_novelty_score(self, state: np.ndarray) -> float:
        """
        Get novelty score for a state (0-1 scale).
        Higher = more novel.
        """
        intrinsic = self.compute_intrinsic_reward(state, update_stats=False)
        
        # Convert to 0-1 scale using sigmoid-like transformation
        novelty = 2.0 / (1.0 + np.exp(-intrinsic)) - 1.0
        return max(0.0, min(1.0, novelty))
    
    def get_stats(self) -> Dict[str, float]:
        """Get curiosity statistics."""
        stats = {
            "total_intrinsic_reward": self.total_intrinsic_reward,
            "training_steps": self.training_steps,
            "reward_mean": self.reward_rms.mean,
            "reward_std": np.sqrt(self.reward_rms.var),
        }
        
        if self.intrinsic_reward_history:
            recent = list(self.intrinsic_reward_history)[-100:]
            stats["recent_avg_intrinsic"] = np.mean(recent)
            stats["recent_max_intrinsic"] = np.max(recent)
        
        if self.novelty_history:
            recent = list(self.novelty_history)[-100:]
            stats["recent_avg_novelty"] = np.mean(recent)
        
        return stats
    
    def save(self, path: str):
        """Save curiosity module state."""
        torch.save({
            "rnd_predictor": self.rnd.predictor.state_dict(),
            "rnd_target": self.rnd.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reward_rms_mean": self.reward_rms.mean,
            "reward_rms_var": self.reward_rms.var,
            "reward_rms_count": self.reward_rms.count,
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "total_intrinsic_reward": self.total_intrinsic_reward,
            "training_steps": self.training_steps,
        }, path)
    
    def load(self, path: str):
        """Load curiosity module state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.rnd.predictor.load_state_dict(checkpoint["rnd_predictor"])
        self.rnd.target.load_state_dict(checkpoint["rnd_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        self.reward_rms.mean = checkpoint["reward_rms_mean"]
        self.reward_rms.var = checkpoint["reward_rms_var"]
        self.reward_rms.count = checkpoint["reward_rms_count"]
        
        self.obs_rms.mean = checkpoint["obs_rms_mean"]
        self.obs_rms.var = checkpoint["obs_rms_var"]
        self.obs_rms.count = checkpoint["obs_rms_count"]
        
        self.total_intrinsic_reward = checkpoint["total_intrinsic_reward"]
        self.training_steps = checkpoint["training_steps"]


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing CuriosityModule...")
    
    curiosity = CuriosityModule(state_dim=64)
    
    # Test single state
    state = np.random.randn(64).astype(np.float32)
    reward = curiosity.compute_intrinsic_reward(state)
    print(f"✓ Single intrinsic reward: {reward:.4f}")
    
    # Test novelty score
    novelty = curiosity.get_novelty_score(state)
    print(f"✓ Novelty score: {novelty:.4f}")
    
    # Test batch
    states = np.random.randn(32, 64).astype(np.float32)
    rewards = curiosity.compute_batch_intrinsic_rewards(states)
    print(f"✓ Batch rewards shape: {rewards.shape}")
    
    # Test training
    for i in range(10):
        train_states = np.random.randn(32, 64).astype(np.float32)
        stats = curiosity.train_step(train_states)
    print(f"✓ Training stats: loss={stats['rnd_loss']:.4f}")
    
    # Test stats
    stats = curiosity.get_stats()
    print(f"✓ Module stats: {stats}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        curiosity.save(f.name)
        curiosity2 = CuriosityModule(state_dim=64)
        curiosity2.load(f.name)
        print("✓ Save/load successful")
    
    print("\n✅ All curiosity tests passed!")
