# rlhf/reward_model.py
"""
Dheera v0.3.0 - RLHF Reward Model
Neural network that learns to predict rewards from human feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
from collections import deque


class RewardNetwork(nn.Module):
    """
    Neural network for reward prediction.
    
    Input: state + action + response_embedding
    Output: scalar reward
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 8,
        response_dim: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.response_dim = response_dim
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, 16)
        
        # Input dimension: state + action_emb + response
        input_dim = state_dim + 16 + response_dim
        
        # MLP layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        response_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch, state_dim)
            action: (batch,) - action indices
            response_embedding: (batch, response_dim)
            
        Returns:
            rewards: (batch, 1)
        """
        # Embed action
        action_emb = self.action_embedding(action)  # (batch, 16)
        
        # Concatenate inputs
        x = torch.cat([state, action_emb, response_embedding], dim=-1)
        
        # Forward through network
        reward = self.network(x)
        
        return reward


class RewardModel:
    """
    Complete reward model for RLHF.
    
    Learns to predict rewards from:
    - Human feedback (++, +, -, --)
    - Preference comparisons
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 8,
        response_dim: int = 384,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.response_dim = response_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = RewardNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            response_dim=response_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=lr,
            weight_decay=0.01,
        )
        
        # Tracking
        self.training_steps = 0
        self.loss_history = deque(maxlen=1000)
    
    def predict_reward(
        self,
        state: np.ndarray,
        action: int,
        response_embedding: np.ndarray,
    ) -> float:
        """
        Predict reward for a single sample.
        
        Args:
            state: State vector
            action: Action taken
            response_embedding: Embedding of response
            
        Returns:
            Predicted reward
        """
        self.network.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = torch.LongTensor([action]).to(self.device)
            response_t = torch.FloatTensor(response_embedding).unsqueeze(0).to(self.device)
            
            reward = self.network(state_t, action_t, response_t)
            return reward.item()
    
    def predict_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        response_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Predict rewards for a batch."""
        self.network.eval()
        
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            responses_t = torch.FloatTensor(response_embeddings).to(self.device)
            
            rewards = self.network(states_t, actions_t, responses_t)
            return rewards.cpu().numpy().flatten()
    
    def train_on_feedback(
        self,
        state: np.ndarray,
        action: int,
        response_embedding: np.ndarray,
        human_reward: float,
    ) -> float:
        """
        Train on direct human feedback.
        
        Args:
            state: State vector
            action: Action taken
            response_embedding: Response embedding
            human_reward: Human-provided reward (-1 to 1)
            
        Returns:
            Training loss
        """
        self.network.train()
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        response_t = torch.FloatTensor(response_embedding).unsqueeze(0).to(self.device)
        target_t = torch.FloatTensor([human_reward]).unsqueeze(1).to(self.device)
        
        # Forward
        predicted = self.network(state_t, action_t, response_t)
        
        # MSE loss
        loss = F.mse_loss(predicted, target_t)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def train_on_preference(
        self,
        state: np.ndarray,
        action_chosen: int,
        response_chosen_emb: np.ndarray,
        action_rejected: int,
        response_rejected_emb: np.ndarray,
        preference_strength: float = 1.0,
    ) -> Tuple[float, bool]:
        """
        Train on preference comparison (Bradley-Terry model).
        
        Human prefers response_chosen over response_rejected.
        
        Args:
            state: Shared state
            action_chosen: Action for chosen response
            response_chosen_emb: Chosen response embedding
            action_rejected: Action for rejected response
            response_rejected_emb: Rejected response embedding
            preference_strength: How strong the preference is (0.5-1.0)
            
        Returns:
            (loss, prediction_correct)
        """
        self.network.train()
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Chosen
        action_chosen_t = torch.LongTensor([action_chosen]).to(self.device)
        response_chosen_t = torch.FloatTensor(response_chosen_emb).unsqueeze(0).to(self.device)
        r_chosen = self.network(state_t, action_chosen_t, response_chosen_t)
        
        # Rejected
        action_rejected_t = torch.LongTensor([action_rejected]).to(self.device)
        response_rejected_t = torch.FloatTensor(response_rejected_emb).unsqueeze(0).to(self.device)
        r_rejected = self.network(state_t, action_rejected_t, response_rejected_t)
        
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(r_chosen - r_rejected) * preference_strength
        
        # Check if prediction is correct
        correct = (r_chosen > r_rejected).item()
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        self.loss_history.append(loss.item())
        
        return loss.item(), correct
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        recent_loss = list(self.loss_history)[-100:] if self.loss_history else [0]
        
        return {
            "training_steps": self.training_steps,
            "avg_loss": np.mean(recent_loss),
            "min_loss": np.min(recent_loss) if recent_loss else 0,
            "device": self.device,
        }
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "response_dim": self.response_dim,
            }
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint["training_steps"]


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing RewardModel...")
    
    model = RewardModel(
        state_dim=64,
        action_dim=8,
        response_dim=384,
    )
    
    print(f"✓ Model created on device: {model.device}")
    
    # Test prediction
    state = np.random.randn(64).astype(np.float32)
    action = 3
    response_emb = np.random.randn(384).astype(np.float32)
    
    reward = model.predict_reward(state, action, response_emb)
    print(f"✓ Predicted reward: {reward:.4f}")
    
    # Test batch prediction
    states = np.random.randn(10, 64).astype(np.float32)
    actions = np.random.randint(0, 8, size=10)
    responses = np.random.randn(10, 384).astype(np.float32)
    
    rewards = model.predict_batch(states, actions, responses)
    print(f"✓ Batch predictions shape: {rewards.shape}")
    
    # Test training on feedback
    loss = model.train_on_feedback(state, action, response_emb, human_reward=0.8)
    print(f"✓ Feedback training loss: {loss:.4f}")
    
    # Test training on preference
    response_chosen = np.random.randn(384).astype(np.float32)
    response_rejected = np.random.randn(384).astype(np.float32)
    
    loss, correct = model.train_on_preference(
        state=state,
        action_chosen=0,
        response_chosen_emb=response_chosen,
        action_rejected=1,
        response_rejected_emb=response_rejected,
    )
    print(f"✓ Preference training loss: {loss:.4f}, correct: {correct}")
    
    # Test stats
    print(f"✓ Stats: {model.get_stats()}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        model.save(f.name)
        model2 = RewardModel(state_dim=64, action_dim=8, response_dim=384)
        model2.load(f.name)
        print(f"✓ Save/load successful, steps: {model2.training_steps}")
    
    print("\n✅ Reward model tests passed!")
