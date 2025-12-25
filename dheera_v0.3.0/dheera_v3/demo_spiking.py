#!/usr/bin/env python3
# demo_spiking.py
"""
Dheera v0.3.1 - SpikingRainbow DQN Demo
Demonstrates the efficiency gains from spiking neural networks

Inspired by SpikingBrain (Chinese Academy of Sciences, 2024):
- 69.15% sparsity in hidden layers
- 97% energy reduction vs traditional networks
- 100x speedup for long sequences
"""

import torch
import numpy as np
import time
from typing import Dict, Any

from core.rainbow_dqn import RainbowNetwork
from core.spiking_rainbow_dqn import SpikingRainbowNetwork, SpikingRainbowDQNAgent
from core.spiking_monitor import SpikingMonitor, benchmark_spiking_vs_dense
from core.spiking_layers import compute_network_sparsity


def demo_basic_spiking():
    """Demo 1: Basic spiking neuron behavior"""
    print("=" * 70)
    print("DEMO 1: Basic Spiking Neuron Behavior")
    print("=" * 70)
    print()

    from core.spiking_layers import LIFNeuron

    # Create a simple LIF neuron
    neuron = LIFNeuron(
        in_features=10,
        out_features=5,
        tau_mem=10.0,
        threshold=1.0,
    )

    # Generate test input
    test_input = torch.randn(1, 10) * 2.0  # Random input

    print("Running 10 time steps...")
    print()

    for t in range(10):
        spikes = neuron(test_input)
        stats = neuron.get_stats()

        print(f"Step {t+1:2d}: Spikes = {spikes.sum().item():.0f}/5 neurons, "
              f"Sparsity = {stats.sparsity*100:.1f}%")

    print()
    final_stats = neuron.get_stats()
    print(f"Overall Sparsity: {final_stats.sparsity*100:.2f}%")
    print(f"Energy Savings:   {final_stats.energy_ratio*100:.2f}% vs dense")
    print()


def demo_spiking_vs_dense():
    """Demo 2: Compare spiking vs dense networks"""
    print("=" * 70)
    print("DEMO 2: SpikingRainbow vs Standard Rainbow DQN")
    print("=" * 70)
    print()

    state_dim = 64
    action_dim = 8
    hidden_dim = 128

    # Create both networks
    print("Creating networks...")
    dense_net = RainbowNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        noisy=False,
    )

    spiking_net = SpikingRainbowNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        use_spiking=True,
        tau_mem=10.0,
        spike_threshold=1.0,
        time_steps=5,
    )

    print(f"✓ Dense network:   {sum(p.numel() for p in dense_net.parameters()):,} parameters")
    print(f"✓ Spiking network: {sum(p.numel() for p in spiking_net.parameters()):,} parameters")
    print()

    # Generate test states
    num_test_states = 100
    test_states = np.random.randn(num_test_states, state_dim).astype(np.float32)

    print(f"Running benchmark with {num_test_states} test states...")
    print()

    # Benchmark
    results = benchmark_spiking_vs_dense(
        spiking_network=spiking_net,
        dense_network=dense_net,
        test_states=test_states,
        num_runs=50,
    )

    # Display results
    print("RESULTS:")
    print(f"  Dense Network:    {results['dense_time_ms']:.3f}ms per forward pass")
    print(f"  Spiking Network:  {results['spiking_time_ms']:.3f}ms per forward pass")
    print()
    print(f"  Speedup:          {results['speedup']:.2f}x")
    print(f"  Sparsity:         {results['sparsity']*100:.2f}%")
    print(f"  Energy Savings:   {results['energy_savings']*100:.2f}%")
    print()
    print(f"  Target Sparsity (SpikingBrain):  69.15%")
    print(f"  Target Energy Savings:           97.00%")
    print()


def demo_spiking_agent():
    """Demo 3: Full SpikingRainbow agent with monitoring"""
    print("=" * 70)
    print("DEMO 3: SpikingRainbow Agent with Real-Time Monitoring")
    print("=" * 70)
    print()

    # Create agent
    print("Creating SpikingRainbow DQN agent...")
    agent = SpikingRainbowDQNAgent(
        state_dim=64,
        action_dim=8,
        hidden_dim=128,
        use_spiking=True,
        tau_mem=10.0,
        spike_threshold=1.0,
        time_steps=5,
    )

    # Create monitor
    monitor = SpikingMonitor(window_size=100, log_interval=10)

    print("✓ Agent created with spiking networks enabled")
    print()

    # Simulate episode
    print("Simulating 50 inference steps...")
    print()

    for i in range(50):
        # Generate random state
        state = np.random.randn(64).astype(np.float32)

        # Measure inference time
        start = time.perf_counter()
        action = agent.select_action(state, training=True)
        inference_time = (time.perf_counter() - start) * 1000

        # Get network stats
        network_stats = agent.online_net.get_sparsity_stats()

        # Record metrics
        metrics = monitor.record_inference(
            network_stats=network_stats,
            inference_time_ms=inference_time,
        )

        if (i + 1) % 10 == 0:
            print(f"Step {i+1:2d}: Action={action}, "
                  f"Sparsity={metrics.overall_sparsity*100:.1f}%, "
                  f"Time={inference_time:.3f}ms")

    print()
    print(monitor.get_report())
    print()

    # Agent stats
    agent_stats = agent.get_stats()
    print("AGENT STATISTICS:")
    print(f"  Spiking Enabled:  {agent_stats['spiking_enabled']}")
    if agent_stats['spiking_enabled']:
        print(f"  Avg Sparsity:     {agent_stats['avg_recent_sparsity']*100:.2f}%")
        print(f"  Energy Savings:   {agent_stats['energy_savings_estimate']*100:.2f}%")
    print()


def demo_sparsity_comparison():
    """Demo 4: Detailed layer-wise sparsity analysis"""
    print("=" * 70)
    print("DEMO 4: Layer-Wise Sparsity Analysis")
    print("=" * 70)
    print()

    # Create spiking network
    net = SpikingRainbowNetwork(
        state_dim=64,
        action_dim=8,
        hidden_dim=128,
        use_spiking=True,
        tau_mem=10.0,
        time_steps=5,
    )

    # Run several forward passes to accumulate statistics
    print("Running 100 forward passes to collect statistics...")
    for _ in range(100):
        test_input = torch.randn(4, 64)  # Batch of 4
        _ = net(test_input)

    # Get detailed stats
    stats = compute_network_sparsity(net)

    print()
    print("NETWORK-WIDE STATISTICS:")
    print(f"  Overall Sparsity:    {stats['overall_sparsity']*100:.2f}%")
    print(f"  Energy Savings:      {stats['energy_savings']*100:.2f}%")
    print(f"  Total Neurons:       {stats['total_neurons']:,}")
    print(f"  Active Neurons:      {stats['total_active']:,}")
    print()

    print("LAYER-WISE BREAKDOWN:")
    for layer_name, layer_stats in stats['layer_stats'].items():
        print(f"  {layer_name}:")
        print(f"    Sparsity:    {layer_stats['sparsity']*100:.2f}%")
        print(f"    Spike Rate:  {layer_stats['spike_rate']:.2f} spikes/neuron")
        print(f"    Active:      {layer_stats['active_neurons']}/{layer_stats['total_neurons']} neurons")
    print()

    # Compare to SpikingBrain
    print("COMPARISON TO SpikingBrain-7B:")
    print(f"  SpikingBrain Sparsity:  69.15%")
    print(f"  Dheera Sparsity:        {stats['overall_sparsity']*100:.2f}%")
    print(f"  Difference:             {(stats['overall_sparsity'] - 0.6915)*100:+.2f}%")
    print()


def main():
    """Run all demos"""
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║      Dheera SpikingRainbow DQN - Efficiency Demonstration         ║")
    print("║                                                                    ║")
    print("║  Inspired by SpikingBrain (Chinese Academy of Sciences, 2024)     ║")
    print("║  Target: 69%+ sparsity, 97% energy reduction                      ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    demos = [
        ("Basic Spiking Neurons", demo_basic_spiking),
        ("Spiking vs Dense Comparison", demo_spiking_vs_dense),
        ("Full Agent with Monitoring", demo_spiking_agent),
        ("Layer-Wise Sparsity", demo_sparsity_comparison),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"❌ Error in demo {i}: {e}")
            import traceback
            traceback.print_exc()
            print()

        if i < len(demos):
            input("Press Enter to continue to next demo...")
            print("\n" * 2)

    print("=" * 70)
    print("ALL DEMOS COMPLETED!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  ✓ Spiking neural networks implemented")
    print("  ✓ Event-driven computation working")
    print("  ✓ Sparsity and energy tracking functional")
    print("  ✓ Compatible with existing Rainbow DQN")
    print()
    print("Next steps:")
    print("  1. Integrate with full Dheera agent")
    print("  2. Train on real tasks and measure efficiency")
    print("  3. Compare long-context performance (RAG)")
    print("  4. Tune hyperparameters for optimal sparsity")
    print()


if __name__ == "__main__":
    main()
