#!/usr/bin/env python3
"""Quick test of spiking implementation"""
import torch
import numpy as np
from core import SpikingRainbowDQNAgent, SpikingMonitor

print("🧠 Testing Dheera SpikingRainbow DQN...")
print()

# Create agent
agent = SpikingRainbowDQNAgent(
    state_dim=64,
    action_dim=8,
    use_spiking=True,
    tau_mem=10.0,
    time_steps=5,
)

# Create monitor
monitor = SpikingMonitor(window_size=20)

# Run 20 inferences
print("Running 20 inference steps...")
for i in range(20):
    state = np.random.randn(64).astype(np.float32)
    action = agent.select_action(state)

    stats = agent.online_net.get_sparsity_stats()
    monitor.record_inference(stats, inference_time_ms=1.0)

# Get stats
stats = agent.get_stats()
summary = monitor.get_summary()

print()
print("✅ TEST RESULTS:")
print(f"  Spiking Enabled: {stats['spiking_enabled']}")
print(f"  Avg Sparsity:    {summary['sparsity']['mean']*100:.1f}%")
print(f"  Target:          {monitor.TARGET_SPARSITY*100:.1f}%")
print(f"  Achievement:     {summary['sparsity']['achievement']*100:.0f}%")
print()

if summary['sparsity']['mean'] > 0.5:
    print("🎉 SUCCESS! Sparsity > 50% achieved!")
else:
    print("⚠️  Sparsity lower than expected (run more steps)")

print()
print("Full stats:")
print(monitor.get_report())
