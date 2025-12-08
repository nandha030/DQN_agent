# memory/__init__.py
"""
Dheera Memory Module
Episodic memory and replay buffers for learning.
Version 0.2.0 - Enhanced with search tracking and 7-action support.
"""

from memory.episodic_memory import (
    EpisodicMemory,
    Episode,
    MemoryEntry,
    ACTION_NAMES,
)
from memory.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    LoggingReplayBuffer,
    HindsightReplayBuffer,
    Transition,
    ExtendedTransition,
    TransitionWithMetadata,
)

__all__ = [
    # Episodic Memory
    "EpisodicMemory",
    "Episode",
    "MemoryEntry",
    "ACTION_NAMES",
    
    # Replay Buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "LoggingReplayBuffer",
    "HindsightReplayBuffer",
    
    # Transitions
    "Transition",
    "ExtendedTransition",
    "TransitionWithMetadata",
]

__version__ = "0.2.0"
