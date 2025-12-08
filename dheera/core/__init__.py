# core/__init__.py
"""
Dheera Core Module
Contains the DQN agent, action space, and state builder.
Version 0.2.0 - Updated with enhanced features and 7-action space.
"""

from core.dqn_agent import (
    TinyDQN,
    RNDModule,
    DQNAgent,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    Transition,
)
from core.action_space import (
    ActionSpace,
    Action,
    ActionType,
)
from core.state_builder import (
    StateBuilder,
    ConversationContext,
)

__all__ = [
    # DQN Agent
    "TinyDQN",
    "RNDModule",
    "DQNAgent",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "Transition",
    
    # Action Space
    "ActionSpace",
    "Action",
    "ActionType",
    
    # State Builder
    "StateBuilder",
    "ConversationContext",
]

# Version info
__version__ = "0.2.0"

# Action IDs for easy access
ACTION_DIRECT_RESPONSE = 0
ACTION_CLARIFY_QUESTION = 1
ACTION_USE_TOOL = 2
ACTION_SEARCH_WEB = 3
ACTION_BREAK_DOWN_TASK = 4
ACTION_REFLECT_AND_REASON = 5
ACTION_DEFER_OR_DECLINE = 6
