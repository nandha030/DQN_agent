# core/__init__.py
"""
Dheera v0.3.1 - Core Package
Rainbow DQN, Curiosity, State Builder, Action Space, Spiking Networks.
"""

from core.rainbow_dqn import RainbowDQNAgent, RainbowNetwork
from core.curiosity_rnd import CuriosityModule, RNDNetwork
from core.state_builder import StateBuilder
from core.action_space import ActionSpace, Action, ActionInfo

# Spiking neural networks (SpikingBrain-inspired)
from core.spiking_layers import (
    LIFNeuron,
    SpikingLinear,
    AdaptiveSpikingLayer,
    compute_network_sparsity,
    SpikeStats,
)
from core.spiking_rainbow_dqn import (
    SpikingRainbowNetwork,
    SpikingRainbowDQNAgent,
    convert_rainbow_to_spiking,
)
from core.spiking_monitor import (
    SpikingMonitor,
    SpikingMetrics,
    benchmark_spiking_vs_dense,
)
from core.spiking_attention import (
    SpikingAttention,
    MultiHeadSpikingAttention,
    SpikingTransformerBlock,
    TemporalSparseMask,
    AttentionStats,
    benchmark_sparse_attention,
)

__all__ = [
    "RainbowDQNAgent",
    "RainbowNetwork",
    "CuriosityModule",
    "RNDNetwork",
    "StateBuilder",
    "ActionSpace",
    "Action",
    "ActionInfo",
    # Spiking
    "LIFNeuron",
    "SpikingLinear",
    "AdaptiveSpikingLayer",
    "compute_network_sparsity",
    "SpikeStats",
    "SpikingRainbowNetwork",
    "SpikingRainbowDQNAgent",
    "convert_rainbow_to_spiking",
    "SpikingMonitor",
    "SpikingMetrics",
    "benchmark_spiking_vs_dense",
    # Spiking Attention
    "SpikingAttention",
    "MultiHeadSpikingAttention",
    "SpikingTransformerBlock",
    "TemporalSparseMask",
    "AttentionStats",
    "benchmark_sparse_attention",
]
