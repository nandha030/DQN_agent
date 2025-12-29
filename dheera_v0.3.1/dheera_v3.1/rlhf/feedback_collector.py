# rlhf/feedback_collector.py
"""
Dheera v0.3.0 - Feedback Collector
Collects and processes human feedback for RLHF.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from rlhf.reward_model import RewardModel
from rlhf.preference_learner import PreferenceLearner, PreferencePair


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    turn_id: int
    episode_id: str
    
    # Context
    user_message: str
    assistant_response: str
    action_id: int
    state_vector: np.ndarray
    response_embedding: np.ndarray
    
    # Feedback
    feedback_type: str  # '++', '+', '-', '--'
    feedback_value: float  # 1.0, 0.5, -0.5, -1.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False


class FeedbackCollector:
    """
    Collects human feedback and coordinates RLHF training.
    
    Handles:
    - Feedback parsing (++, +, -, --)
    - Feedback storage
    - Triggering reward model updates
    - Creating preference pairs
    """
    
    FEEDBACK_MAP = {
        "++": 1.0,
        "+": 0.5,
        "-": -0.5,
        "--": -1.0,
    }
    
    def __init__(
        self,
        reward_model: RewardModel,
        preference_learner: PreferenceLearner,
        db_manager: Optional[Any] = None,
        train_every_n_feedback: int = 5,
        min_preferences_for_training: int = 10,
    ):
        self.reward_model = reward_model
        self.preference_learner = preference_learner
        self.db = db_manager
        self.train_every_n = train_every_n_feedback
        self.min_preferences = min_preferences_for_training
        
        # Feedback buffer
        self.feedback_buffer: deque = deque(maxlen=1000)
        self.recent_positive: List[FeedbackEntry] = []
        self.recent_negative: List[FeedbackEntry] = []
        
        # Statistics
        self.total_feedback = 0
        self.positive_count = 0
        self.negative_count = 0
        self.feedback_since_train = 0
    
    def parse_feedback(self, feedback_str: str) -> Optional[float]:
        """Parse feedback string to value."""
        feedback_str = feedback_str.strip()
        return self.FEEDBACK_MAP.get(feedback_str)
    
    def is_feedback(self, text: str) -> bool:
        """Check if text is feedback."""
        return text.strip() in self.FEEDBACK_MAP
    
    def collect(
        self,
        turn_id: int,
        episode_id: str,
        user_message: str,
        assistant_response: str,
        action_id: int,
        state_vector: np.ndarray,
        response_embedding: np.ndarray,
        feedback_str: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Collect and process feedback.
        
        Args:
            turn_id: Turn identifier
            episode_id: Episode identifier
            user_message: Original user message
            assistant_response: Assistant's response
            action_id: Action taken
            state_vector: State when action was taken
            response_embedding: Embedding of response
            feedback_str: Feedback string (++, +, -, --)
            
        Returns:
            (feedback_value, processing_info)
        """
        feedback_value = self.parse_feedback(feedback_str)
        
        if feedback_value is None:
            return 0.0, {"error": "Invalid feedback string"}
        
        # Create entry
        entry = FeedbackEntry(
            turn_id=turn_id,
            episode_id=episode_id,
            user_message=user_message,
            assistant_response=assistant_response,
            action_id=action_id,
            state_vector=state_vector,
            response_embedding=response_embedding,
            feedback_type=feedback_str.strip(),
            feedback_value=feedback_value,
        )
        
        # Store
        self.feedback_buffer.append(entry)
        self.total_feedback += 1
        self.feedback_since_train += 1
        
        # Track positive/negative
        if feedback_value > 0:
            self.positive_count += 1
            self.recent_positive.append(entry)
            if len(self.recent_positive) > 20:
                self.recent_positive.pop(0)
        else:
            self.negative_count += 1
            self.recent_negative.append(entry)
            if len(self.recent_negative) > 20:
                self.recent_negative.pop(0)
        
        # Train reward model directly on this feedback
        loss = self.reward_model.train_on_feedback(
            state=state_vector,
            action=action_id,
            response_embedding=response_embedding,
            human_reward=feedback_value,
        )
        
        # Create preference pairs
        preferences_created = self._create_preference_pairs(entry)
        
        # Maybe train preference learner
        train_stats = None
        if self.feedback_since_train >= self.train_every_n:
            if self.preference_learner.total_preferences >= self.min_preferences:
                train_stats = self.preference_learner.train_step()
                self.feedback_since_train = 0
        
        entry.processed = True
        
        return feedback_value, {
            "feedback_value": feedback_value,
            "reward_loss": loss,
            "preferences_created": preferences_created,
            "train_stats": train_stats,
        }
    
    def _create_preference_pairs(self, entry: FeedbackEntry) -> int:
        """Create preference pairs from feedback entry."""
        created = 0
        
        if entry.feedback_value > 0:
            # Positive feedback - pair with recent negative
            for neg_entry in self.recent_negative[-3:]:
                preference = PreferencePair(
                    id=f"pref_{entry.turn_id}_{neg_entry.turn_id}",
                    state=entry.state_vector,
                    action_chosen=entry.action_id,
                    response_chosen_emb=entry.response_embedding,
                    response_chosen_text=entry.assistant_response,
                    action_rejected=neg_entry.action_id,
                    response_rejected_emb=neg_entry.response_embedding,
                    response_rejected_text=neg_entry.assistant_response,
                    preference_strength=abs(entry.feedback_value),
                    source="human",
                )
                self.preference_learner.add_preference(preference)
                created += 1
        
        else:
            # Negative feedback - pair with recent positive
            for pos_entry in self.recent_positive[-3:]:
                preference = PreferencePair(
                    id=f"pref_{pos_entry.turn_id}_{entry.turn_id}",
                    state=entry.state_vector,
                    action_chosen=pos_entry.action_id,
                    response_chosen_emb=pos_entry.response_embedding,
                    response_chosen_text=pos_entry.assistant_response,
                    action_rejected=entry.action_id,
                    response_rejected_emb=entry.response_embedding,
                    response_rejected_text=entry.assistant_response,
                    preference_strength=abs(entry.feedback_value),
                    source="human",
                )
                self.preference_learner.add_preference(preference)
                created += 1
        
        return created
    
    def get_reward_bonus(
        self,
        state: np.ndarray,
        action: int,
        response_embedding: np.ndarray,
        base_reward: float = 0.0,
        blend_factor: float = 0.3,
    ) -> float:
        """
        Get reward with RLHF model bonus.
        
        Args:
            state: Current state
            action: Action taken
            response_embedding: Response embedding
            base_reward: Base reward (from rules/heuristics)
            blend_factor: How much to weight model prediction (0-1)
            
        Returns:
            Combined reward
        """
        if self.preference_learner.total_preferences < self.min_preferences:
            # Not enough data, return base reward
            return base_reward
        
        # Get model prediction
        model_reward = self.reward_model.predict_reward(
            state, action, response_embedding
        )
        
        # Blend
        combined = (1 - blend_factor) * base_reward + blend_factor * model_reward
        
        return combined
    
    def get_action_preferences(
        self,
        state: np.ndarray,
        response_embeddings: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        """
        Get reward model preferences for different actions.
        
        Args:
            state: Current state
            response_embeddings: Dict mapping action -> response embedding
            
        Returns:
            Dict mapping action -> predicted reward
        """
        preferences = {}
        
        for action, emb in response_embeddings.items():
            reward = self.reward_model.predict_reward(state, action, emb)
            preferences[action] = reward
        
        return preferences
    
    def force_train(self, epochs: int = 1) -> List[Dict[str, float]]:
        """Force training of preference learner."""
        results = []
        
        for _ in range(epochs):
            if self.preference_learner.total_preferences >= self.min_preferences:
                stats = self.preference_learner.train_epoch()
                results.append(stats)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "total_feedback": self.total_feedback,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_ratio": self.positive_count / max(self.total_feedback, 1),
            "buffer_size": len(self.feedback_buffer),
            "preference_learner": self.preference_learner.get_stats(),
            "reward_model": self.reward_model.get_stats(),
        }
    
    def save(self, path: str):
        """Save the feedback system state."""
        self.reward_model.save(f"{path}_reward_model.pt")
    
    def load(self, path: str):
        """Load the feedback system state."""
        self.reward_model.load(f"{path}_reward_model.pt")


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing FeedbackCollector...")
    
    # Setup
    reward_model = RewardModel(state_dim=64, action_dim=8, response_dim=384)
    preference_learner = PreferenceLearner(reward_model)
    collector = FeedbackCollector(
        reward_model=reward_model,
        preference_learner=preference_learner,
        train_every_n_feedback=3,
        min_preferences_for_training=5,
    )
    
    # Test feedback parsing
    assert collector.parse_feedback("++") == 1.0
    assert collector.parse_feedback("+") == 0.5
    assert collector.parse_feedback("-") == -0.5
    assert collector.parse_feedback("--") == -1.0
    print("✓ Feedback parsing works")
    
    # Collect some feedback
    for i in range(10):
        feedback = "++" if i % 2 == 0 else "--"
        value, info = collector.collect(
            turn_id=i,
            episode_id="test_ep",
            user_message=f"Question {i}",
            assistant_response=f"Answer {i}",
            action_id=i % 8,
            state_vector=np.random.randn(64).astype(np.float32),
            response_embedding=np.random.randn(384).astype(np.float32),
            feedback_str=feedback,
        )
        
        if i == 0:
            print(f"✓ First feedback: value={value}, info keys={list(info.keys())}")
    
    print(f"✓ Collected {collector.total_feedback} feedback entries")
    print(f"✓ Created {collector.preference_learner.total_preferences} preferences")
    
    # Test reward bonus
    state = np.random.randn(64).astype(np.float32)
    emb = np.random.randn(384).astype(np.float32)
    bonus = collector.get_reward_bonus(state, 0, emb, base_reward=0.5)
    print(f"✓ Reward with bonus: {bonus:.4f}")
    
    # Test action preferences
    embs = {i: np.random.randn(384).astype(np.float32) for i in range(4)}
    prefs = collector.get_action_preferences(state, embs)
    print(f"✓ Action preferences: {prefs}")
    
    # Force train
    results = collector.force_train(epochs=2)
    print(f"✓ Force train results: {len(results)} epochs")
    
    # Stats
    stats = collector.get_stats()
    print(f"✓ Stats: positive_ratio={stats['positive_ratio']:.2f}")
    
    print("\n✅ Feedback collector tests passed!")
