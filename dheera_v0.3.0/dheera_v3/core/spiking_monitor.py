# core/spiking_monitor.py
"""
Dheera v0.3.1 - Spiking Network Performance Monitor
Track efficiency metrics inspired by SpikingBrain benchmarks

Monitors:
- Sparsity (target: 69%+ like SpikingBrain-7B)
- Energy efficiency (target: 97% reduction)
- Spike rates per layer
- Inference speedup vs dense networks
- Temporal dynamics

Provides real-time visualization and logging compatible
with Dheera's existing logging infrastructure.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import json


@dataclass
class SpikingMetrics:
    """Container for spiking network performance metrics"""

    # Sparsity metrics (SpikingBrain: 69.15%)
    overall_sparsity: float = 0.0
    layer_sparsity: Dict[str, float] = field(default_factory=dict)

    # Energy metrics (SpikingBrain: 97% reduction)
    energy_savings_estimate: float = 0.0
    active_neuron_ratio: float = 0.0

    # Performance metrics
    inference_time_ms: float = 0.0
    speedup_vs_dense: float = 1.0

    # Spike statistics
    avg_spike_rate: float = 0.0
    spike_rate_per_layer: Dict[str, float] = field(default_factory=dict)

    # Temporal dynamics
    membrane_dynamics: Dict[str, Any] = field(default_factory=dict)

    # Comparison to SpikingBrain targets
    sparsity_vs_target: float = 0.0  # Difference from 69.15%
    energy_vs_target: float = 0.0  # Difference from 97%

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "sparsity": {
                "overall": self.overall_sparsity,
                "layers": self.layer_sparsity,
                "vs_target": self.sparsity_vs_target,
            },
            "energy": {
                "savings_estimate": self.energy_savings_estimate,
                "active_ratio": self.active_neuron_ratio,
                "vs_target": self.energy_vs_target,
            },
            "performance": {
                "inference_ms": self.inference_time_ms,
                "speedup": self.speedup_vs_dense,
            },
            "spikes": {
                "avg_rate": self.avg_spike_rate,
                "per_layer": self.spike_rate_per_layer,
            },
        }


class SpikingMonitor:
    """
    Real-time monitor for spiking network efficiency.

    Tracks performance metrics and compares against
    SpikingBrain benchmarks (69.15% sparsity, 97% energy reduction).
    """

    # SpikingBrain benchmark targets
    TARGET_SPARSITY = 0.6915  # 69.15% from paper
    TARGET_ENERGY_SAVINGS = 0.97  # 97% reduction

    def __init__(
        self,
        window_size: int = 100,
        log_interval: int = 10,
        enable_detailed_logging: bool = False,
    ):
        """
        Args:
            window_size: Moving average window for metrics
            log_interval: Log metrics every N updates
            enable_detailed_logging: Log per-layer details
        """
        self.window_size = window_size
        self.log_interval = log_interval
        self.enable_detailed = enable_detailed_logging

        # Metric histories
        self.sparsity_history = deque(maxlen=window_size)
        self.energy_history = deque(maxlen=window_size)
        self.inference_time_history = deque(maxlen=window_size)

        # Counters
        self.update_count = 0
        self.total_inferences = 0

        # Comparison metrics
        self.dense_baseline_time = None

    def record_inference(
        self,
        network_stats: Dict[str, Any],
        inference_time_ms: float,
        dense_baseline_ms: Optional[float] = None,
    ) -> SpikingMetrics:
        """
        Record metrics from a single inference.

        Args:
            network_stats: Dict from compute_network_sparsity()
            inference_time_ms: Inference time in milliseconds
            dense_baseline_ms: Baseline dense network time (optional)

        Returns:
            SpikingMetrics with current statistics
        """
        self.total_inferences += 1
        self.update_count += 1

        # Extract core metrics
        overall_sparsity = network_stats.get("overall_sparsity", 0.0)
        energy_savings = network_stats.get("energy_savings", 0.0)

        # Record histories
        self.sparsity_history.append(overall_sparsity)
        self.energy_history.append(energy_savings)
        self.inference_time_history.append(inference_time_ms)

        # Update baseline if provided
        if dense_baseline_ms is not None:
            self.dense_baseline_time = dense_baseline_ms

        # Compute speedup
        speedup = 1.0
        if self.dense_baseline_time is not None and self.dense_baseline_time > 0:
            speedup = self.dense_baseline_time / inference_time_ms

        # Layer-wise metrics
        layer_stats = network_stats.get("layer_stats", {})
        layer_sparsity = {}
        spike_rates = {}

        for layer_name, stats in layer_stats.items():
            layer_sparsity[layer_name] = stats.get("sparsity", 0.0)
            spike_rates[layer_name] = stats.get("spike_rate", 0.0)

        # Compute average spike rate
        avg_spike_rate = np.mean(list(spike_rates.values())) if spike_rates else 0.0

        # Active neuron ratio (1 - sparsity)
        active_ratio = 1.0 - overall_sparsity

        # Compare to SpikingBrain targets
        sparsity_vs_target = overall_sparsity - self.TARGET_SPARSITY
        energy_vs_target = energy_savings - self.TARGET_ENERGY_SAVINGS

        # Create metrics object
        metrics = SpikingMetrics(
            overall_sparsity=overall_sparsity,
            layer_sparsity=layer_sparsity,
            energy_savings_estimate=energy_savings,
            active_neuron_ratio=active_ratio,
            inference_time_ms=inference_time_ms,
            speedup_vs_dense=speedup,
            avg_spike_rate=avg_spike_rate,
            spike_rate_per_layer=spike_rates,
            sparsity_vs_target=sparsity_vs_target,
            energy_vs_target=energy_vs_target,
        )

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics over monitoring window"""
        if not self.sparsity_history:
            return {"status": "no_data"}

        summary = {
            "total_inferences": self.total_inferences,
            "sparsity": {
                "current": self.sparsity_history[-1],
                "mean": np.mean(self.sparsity_history),
                "std": np.std(self.sparsity_history),
                "target": self.TARGET_SPARSITY,
                "achievement": np.mean(self.sparsity_history) / self.TARGET_SPARSITY,
            },
            "energy": {
                "current": self.energy_history[-1] if self.energy_history else 0.0,
                "mean": np.mean(self.energy_history) if self.energy_history else 0.0,
                "target": self.TARGET_ENERGY_SAVINGS,
                "achievement": (np.mean(self.energy_history) / self.TARGET_ENERGY_SAVINGS
                               if self.energy_history else 0.0),
            },
            "performance": {
                "avg_inference_ms": np.mean(self.inference_time_history),
                "std_inference_ms": np.std(self.inference_time_history),
                "baseline_ms": self.dense_baseline_time,
            },
        }

        return summary

    def get_report(self) -> str:
        """
        Generate human-readable performance report.

        Compares against SpikingBrain benchmarks.
        """
        summary = self.get_summary()

        if summary.get("status") == "no_data":
            return "⚡ Spiking Monitor: No data collected yet"

        sparsity_stats = summary["sparsity"]
        energy_stats = summary["energy"]
        perf_stats = summary["performance"]

        report = [
            "=" * 60,
            "⚡ SpikingRainbow DQN Performance Report",
            "=" * 60,
            "",
            "📊 SPARSITY METRICS",
            f"  Current:     {sparsity_stats['current']*100:.2f}%",
            f"  Average:     {sparsity_stats['mean']*100:.2f}% ± {sparsity_stats['std']*100:.2f}%",
            f"  Target:      {self.TARGET_SPARSITY*100:.2f}% (SpikingBrain-7B)",
            f"  Achievement: {sparsity_stats['achievement']*100:.1f}%",
            "",
            "⚡ ENERGY EFFICIENCY",
            f"  Savings:     {energy_stats['mean']*100:.2f}%",
            f"  Target:      {self.TARGET_ENERGY_SAVINGS*100:.2f}% (SpikingBrain)",
            f"  Achievement: {energy_stats['achievement']*100:.1f}%",
            "",
            "⏱️  PERFORMANCE",
            f"  Avg Inference: {perf_stats['avg_inference_ms']:.2f}ms",
            f"  Total Calls:   {self.total_inferences}",
            "",
        ]

        if perf_stats["baseline_ms"]:
            speedup = perf_stats["baseline_ms"] / perf_stats["avg_inference_ms"]
            report.extend([
                f"  Dense Baseline: {perf_stats['baseline_ms']:.2f}ms",
                f"  Speedup:        {speedup:.2f}x",
                "",
            ])

        # Status indicator
        if sparsity_stats['mean'] >= 0.6:
            status = "🟢 EXCELLENT (>60% sparsity)"
        elif sparsity_stats['mean'] >= 0.4:
            status = "🟡 GOOD (40-60% sparsity)"
        else:
            status = "🔴 LOW (<40% sparsity)"

        report.extend([
            f"Status: {status}",
            "=" * 60,
        ])

        return "\n".join(report)

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        summary = self.get_summary()

        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "histories": {
                "sparsity": list(self.sparsity_history),
                "energy": list(self.energy_history),
                "inference_time": list(self.inference_time_history),
            },
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ==================== Benchmark Utilities ====================

def benchmark_spiking_vs_dense(
    spiking_network,
    dense_network,
    test_states: np.ndarray,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark spiking vs dense network on same inputs.

    Args:
        spiking_network: SpikingRainbowNetwork instance
        dense_network: Standard RainbowNetwork instance
        test_states: Test state inputs [N, state_dim]
        num_runs: Number of benchmark runs

    Returns:
        Comparison metrics
    """
    import torch

    device = next(spiking_network.parameters()).device
    test_tensor = torch.FloatTensor(test_states).to(device)

    # Warm-up
    for _ in range(10):
        _ = spiking_network(test_tensor)
        _ = dense_network(test_tensor)

    # Benchmark spiking
    spiking_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = spiking_network(test_tensor)
        spiking_times.append((time.perf_counter() - start) * 1000)

    # Benchmark dense
    dense_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = dense_network(test_tensor)
        dense_times.append((time.perf_counter() - start) * 1000)

    # Compute stats
    spiking_avg = np.mean(spiking_times)
    dense_avg = np.mean(dense_times)
    speedup = dense_avg / spiking_avg

    # Get sparsity
    from core.spiking_layers import compute_network_sparsity
    sparsity_stats = compute_network_sparsity(spiking_network)

    results = {
        "spiking_time_ms": spiking_avg,
        "dense_time_ms": dense_avg,
        "speedup": speedup,
        "sparsity": sparsity_stats["overall_sparsity"],
        "energy_savings": sparsity_stats["energy_savings"],
        "interpretation": (
            f"Spiking network is {speedup:.2f}x "
            f"{'faster' if speedup > 1 else 'slower'} than dense, "
            f"with {sparsity_stats['overall_sparsity']*100:.1f}% sparsity"
        ),
    }

    return results
