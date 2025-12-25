# rlhf/preference_learner.py
"""
Dheera v0.3.0 - Preference Learner
Learns from pairwise preferences using Bradley-Terry model.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import random

from rlhf.reward_model import RewardModel


@dataclass
class PreferencePair:
    """A preference comparison pair."""
    id: str
    state: np.ndarray
    
    # Chosen (preferred) response
    action_chosen: int
    response_chosen_emb: np.ndarray
    response_chosen_text: str
    
    # Rejected response
    action_rejected: int
    response_rejected_emb: np.ndarray
    response_rejected_text: str
    
    # Metadata
    preference_strength: float = 1.0  # 0.5 = slight, 1.0 = strong
    source: str = "human"  # 'human', 'synthetic', 'model'
    created_at: datetime = None
    used_in_training: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PreferenceLearner:
    """
    Learns reward model from preference comparisons.
    
    Implements:
    - Bradley-Terry preference learning
    - Active learning for efficient data collection
    - Synthetic preference generation
    """
    
    def __init__(
        self,
        reward_model: RewardModel,
        db_manager: Optional[Any] = None,
    ):
        self.reward_model = reward_model
        self.db = db_manager
        
        # In-memory preference buffer (for when no DB)
        self.preferences: List[PreferencePair] = []
        
        # Statistics
        self.total_preferences = 0
        self.training_epochs = 0
        self.accuracy_history = []
    
    def add_preference(self, preference: PreferencePair):
        """Add a preference pair to the buffer."""
        self.preferences.append(preference)
        self.total_preferences += 1
        
        # Store in database if available
        if self.db:
            self.db.store_preference(
                turn_id_chosen=0,  # Would need actual turn IDs
                turn_id_rejected=0,
                user_message="",
                response_chosen=preference.response_chosen_text,
                response_rejected=preference.response_rejected_text,
                state_vector=preference.state,
                preference_strength=preference.preference_strength,
                source=preference.source,
            )
    
    def create_preference_from_feedback(
        self,
        state: np.ndarray,
        current_action: int,
        current_response_emb: np.ndarray,
        current_response_text: str,
        feedback: float,  # Positive or negative
        recent_turns: List[Dict],  # Recent conversation turns
    ) -> Optional[PreferencePair]:
        """
        Create preference pair from human feedback.
        
        If feedback is positive, pair with a recent negative response.
        If feedback is negative, pair with a recent positive response.
        """
        # Find contrasting response
        contrasting = None
        for turn in recent_turns:
            turn_feedback = turn.get("human_feedback", 0)
            
            if feedback > 0 and turn_feedback < 0:
                # Current is good, find a bad one
                contrasting = turn
                break
            elif feedback < 0 and turn_feedback > 0:
                # Current is bad, find a good one
                contrasting = turn
                break
        
        if contrasting is None:
            return None
        
        # Create preference pair
        if feedback > 0:
            # Current is chosen
            preference = PreferencePair(
                id=f"pref_{self.total_preferences}",
                state=state,
                action_chosen=current_action,
                response_chosen_emb=current_response_emb,
                response_chosen_text=current_response_text,
                action_rejected=contrasting.get("action_id", 0),
                response_rejected_emb=contrasting.get("response_emb", np.zeros(384)),
                response_rejected_text=contrasting.get("assistant_response", ""),
                preference_strength=abs(feedback),
            )
        else:
            # Contrasting is chosen
            preference = PreferencePair(
                id=f"pref_{self.total_preferences}",
                state=state,
                action_chosen=contrasting.get("action_id", 0),
                response_chosen_emb=contrasting.get("response_emb", np.zeros(384)),
                response_chosen_text=contrasting.get("assistant_response", ""),
                action_rejected=current_action,
                response_rejected_emb=current_response_emb,
                response_rejected_text=current_response_text,
                preference_strength=abs(feedback),
            )
        
        self.add_preference(preference)
        return preference
    
    def train_step(self, batch_size: int = 16) -> Dict[str, float]:
        """
        Train reward model on a batch of preferences.
        
        Returns:
            Training statistics
        """
        if len(self.preferences) < batch_size:
            return {"error": "Not enough preferences"}
        
        # Sample batch
        batch = random.sample(self.preferences, min(batch_size, len(self.preferences)))
        
        total_loss = 0.0
        correct = 0
        
        for pref in batch:
            loss, is_correct = self.reward_model.train_on_preference(
                state=pref.state,
                action_chosen=pref.action_chosen,
                response_chosen_emb=pref.response_chosen_emb,
                action_rejected=pref.action_rejected,
                response_rejected_emb=pref.response_rejected_emb,
                preference_strength=pref.preference_strength,
            )
            
            total_loss += loss
            if is_correct:
                correct += 1
            
            pref.used_in_training = True
        
        accuracy = correct / len(batch)
        self.accuracy_history.append(accuracy)
        
        return {
            "loss": total_loss / len(batch),
            "accuracy": accuracy,
            "batch_size": len(batch),
        }
    
    def train_epoch(self, batch_size: int = 16) -> Dict[str, float]:
        """Train on all preferences for one epoch."""
        if not self.preferences:
            return {"error": "No preferences"}
        
        # Shuffle preferences
        shuffled = list(self.preferences)
        random.shuffle(shuffled)
        
        total_loss = 0.0
        total_correct = 0
        num_batches = 0
        
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            
            for pref in batch:
                loss, is_correct = self.reward_model.train_on_preference(
                    state=pref.state,
                    action_chosen=pref.action_chosen,
                    response_chosen_emb=pref.response_chosen_emb,
                    action_rejected=pref.action_rejected,
                    response_rejected_emb=pref.response_rejected_emb,
                    preference_strength=pref.preference_strength,
                )
                
                total_loss += loss
                if is_correct:
                    total_correct += 1
            
            num_batches += 1
        
        self.training_epochs += 1
        accuracy = total_correct / len(shuffled)
        self.accuracy_history.append(accuracy)
        
        return {
            "epoch": self.training_epochs,
            "loss": total_loss / len(shuffled),
            "accuracy": accuracy,
            "num_preferences": len(shuffled),
        }
    
    def generate_synthetic_preference(
        self,
        state: np.ndarray,
        actions: List[int],
        response_embeddings: List[np.ndarray],
        response_texts: List[str],
    ) -> Optional[PreferencePair]:
        """
        Generate synthetic preference using current reward model.
        
        Useful for data augmentation when human feedback is limited.
        """
        if len(actions) < 2:
            return None
        
        # Get reward predictions
        rewards = []
        for action, emb in zip(actions, response_embeddings):
            r = self.reward_model.predict_reward(state, action, emb)
            rewards.append(r)
        
        # Find best and worst
        best_idx = np.argmax(rewards)
        worst_idx = np.argmin(rewards)
        
        if best_idx == worst_idx:
            return None
        
        # Create preference
        preference = PreferencePair(
            id=f"synth_{self.total_preferences}",
            state=state,
            action_chosen=actions[best_idx],
            response_chosen_emb=response_embeddings[best_idx],
            response_chosen_text=response_texts[best_idx],
            action_rejected=actions[worst_idx],
            response_rejected_emb=response_embeddings[worst_idx],
            response_rejected_text=response_texts[worst_idx],
            preference_strength=0.5,  # Lower confidence for synthetic
            source="synthetic",
        )
        
        self.add_preference(preference)
        return preference
    
    def get_uncertainty_samples(self, n: int = 10) -> List[PreferencePair]:
        """
        Get preferences where model is most uncertain.
        Useful for active learning.
        """
        uncertainties = []
        
        for pref in self.preferences:
            r_chosen = self.reward_model.predict_reward(
                pref.state, pref.action_chosen, pref.response_chosen_emb
            )
            r_rejected = self.reward_model.predict_reward(
                pref.state, pref.action_rejected, pref.response_rejected_emb
            )
            
            # Uncertainty = how close the rewards are
            uncertainty = 1.0 / (1.0 + abs(r_chosen - r_rejected))
            uncertainties.append((pref, uncertainty))
        
        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        return [p for p, _ in uncertainties[:n]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        recent_acc = self.accuracy_history[-100:] if self.accuracy_history else [0]
        
        return {
            "total_preferences": self.total_preferences,
            "buffer_size": len(self.preferences),
            "training_epochs": self.training_epochs,
            "recent_accuracy": np.mean(recent_acc),
            "human_preferences": sum(1 for p in self.preferences if p.source == "human"),
            "synthetic_preferences": sum(1 for p in self.preferences if p.source == "synthetic"),
        }


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing PreferenceLearner...")
    
    # Create reward model and learner
    reward_model = RewardModel(state_dim=64, action_dim=8, response_dim=384)
    learner = PreferenceLearner(reward_model)
    
    # Add some preferences
    for i in range(20):
        pref = PreferencePair(
            id=f"test_{i}",
            state=np.random.randn(64).astype(np.float32),
            action_chosen=0,
            response_chosen_emb=np.random.randn(384).astype(np.float32),
            response_chosen_text="Good response",
            action_rejected=1,
            response_rejected_emb=np.random.randn(384).astype(np.float32),
            response_rejected_text="Bad response",
        )
        learner.add_preference(pref)
    
    print(f"✓ Added {learner.total_preferences} preferences")
    
    # Train step
    stats = learner.train_step(batch_size=8)
    print(f"✓ Train step: loss={stats['loss']:.4f}, accuracy={stats['accuracy']:.2f}")
    
    # Train epoch
    stats = learner.train_epoch(batch_size=8)
    print(f"✓ Train epoch: loss={stats['loss']:.4f}, accuracy={stats['accuracy']:.2f}")
    
    # Generate synthetic preference
    state = np.random.randn(64).astype(np.float32)
    actions = [0, 1, 2]
    embs = [np.random.randn(384).astype(np.float32) for _ in range(3)]
    texts = ["Response A", "Response B", "Response C"]
    
    synth = learner.generate_synthetic_preference(state, actions, embs, texts)
    print(f"✓ Generated synthetic preference: {synth is not None}")
    
    # Get uncertain samples
    uncertain = learner.get_uncertainty_samples(n=5)
    print(f"✓ Got {len(uncertain)} uncertain samples")
    
    # Stats
    print(f"✓ Stats: {learner.get_stats()}")
    
    print("\n✅ Preference learner tests passed!")
