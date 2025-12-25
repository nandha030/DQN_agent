# rlhf/__init__.py
"""
Dheera RLHF Module

Contains:
- Reward Model
- Preference Learner
- Feedback Collector
"""

from rlhf.reward_model import RewardModel, RewardNetwork
from rlhf.preference_learner import PreferenceLearner, PreferencePair
from rlhf.feedback_collector import FeedbackCollector, FeedbackEntry

__all__ = [
    # Reward Model
    "RewardModel",
    "RewardNetwork",
    
    # Preference Learning
    "PreferenceLearner",
    "PreferencePair",
    
    # Feedback
    "FeedbackCollector",
    "FeedbackEntry",
]
