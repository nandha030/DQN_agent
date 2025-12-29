#!/usr/bin/env python3
"""
Dheera v0.3.1 - System Profiler & Auto-Configuration
Detects hardware capabilities and recommends optimal Dheera settings
"""

import platform
import subprocess
import json
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class HardwareProfile:
    """Hardware capabilities detected"""
    # CPU
    cpu_count: int
    cpu_brand: str
    cpu_freq_mhz: float

    # Memory
    total_ram_gb: float
    available_ram_gb: float

    # GPU
    has_gpu: bool

    # OS
    os_name: str
    os_version: str

    # Optional GPU fields
    gpu_type: Optional[str] = None  # cuda, mps, rocm, cpu
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None

    # Performance tier
    tier: str = "unknown"  # high, medium, low


@dataclass
class RecommendedConfig:
    """Recommended Dheera configuration based on hardware"""
    # SLM settings
    slm_timeout: int
    slm_max_tokens: int
    slm_model: str

    # DQN settings
    dqn_batch_size: int
    dqn_train_every: int

    # RAG settings
    rag_n_results: int
    rag_max_context_tokens: int

    # Performance profile
    profile_name: str
    reasoning: str


class SystemProfiler:
    """Detects system capabilities and recommends optimal configuration"""

    def __init__(self):
        self.hardware: Optional[HardwareProfile] = None
        self.ollama_models: List[str] = []

    def profile_system(self) -> HardwareProfile:
        """Detect hardware capabilities"""
        print("🔍 Profiling system hardware...")

        # CPU detection
        cpu_count = os.cpu_count() or 1
        cpu_freq_mhz = self._get_cpu_freq()
        cpu_brand = self._get_cpu_brand()

        # Memory detection
        total_ram_gb, available_ram_gb = self._get_memory_info()

        # GPU detection
        gpu_info = self._detect_gpu()

        # OS detection
        os_name = platform.system()
        os_version = platform.release()

        # Determine performance tier
        tier = self._calculate_tier(
            cpu_count, cpu_freq_mhz, total_ram_gb, gpu_info["has_gpu"]
        )

        profile = HardwareProfile(
            cpu_count=cpu_count,
            cpu_brand=cpu_brand,
            cpu_freq_mhz=cpu_freq_mhz,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            has_gpu=gpu_info["has_gpu"],
            gpu_type=gpu_info.get("type"),
            gpu_name=gpu_info.get("name"),
            gpu_vram_gb=gpu_info.get("vram_gb"),
            os_name=os_name,
            os_version=os_version,
            tier=tier,
        )

        self.hardware = profile
        return profile

    def _get_cpu_freq(self) -> float:
        """Get CPU frequency in MHz"""
        try:
            if platform.system() == "Darwin":  # macOS
                cmd = "sysctl -n hw.cpufrequency"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Returns Hz, convert to MHz
                    return float(result.stdout.strip()) / 1_000_000
        except Exception:
            pass
        return 2400.0  # Default assumption

    def _get_memory_info(self) -> tuple:
        """Get total and available RAM in GB"""
        try:
            if platform.system() == "Darwin":  # macOS
                # Total RAM
                cmd = "sysctl -n hw.memsize"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                total_bytes = int(result.stdout.strip())
                total_gb = total_bytes / (1024**3)

                # Available RAM (rough estimate)
                cmd = "vm_stat | grep 'Pages free' | awk '{print $3}' | sed 's/\\.//'"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    free_pages = int(result.stdout.strip())
                    # macOS page size is 4096 bytes
                    available_gb = (free_pages * 4096) / (1024**3)
                else:
                    available_gb = total_gb * 0.5  # Assume 50% available

                return total_gb, available_gb

            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                    mem_info = {}
                    for line in lines:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = int(parts[1].strip().split()[0])
                            mem_info[key] = value

                    total_gb = mem_info.get("MemTotal", 0) / (1024**2)
                    available_gb = mem_info.get("MemAvailable", mem_info.get("MemFree", 0)) / (1024**2)
                    return total_gb, available_gb
        except Exception:
            pass

        # Default fallback
        return 16.0, 8.0

    def _get_cpu_brand(self) -> str:
        """Get CPU brand name"""
        try:
            if platform.system() == "Darwin":  # macOS
                cmd = "sysctl -n machdep.cpu.brand_string"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                return result.stdout.strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Windows":
                cmd = "wmic cpu get name"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split("\n")
                return lines[1] if len(lines) > 1 else "Unknown"
        except Exception:
            pass
        return platform.processor() or "Unknown"

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU availability and type"""
        gpu_info = {"has_gpu": False}

        # Try NVIDIA GPU (CUDA)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(",")
                    gpu_info = {
                        "has_gpu": True,
                        "type": "cuda",
                        "name": parts[0].strip(),
                        "vram_gb": float(parts[1].strip().split()[0]) / 1024 if len(parts) > 1 else None,
                    }
                    return gpu_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try AMD GPU (ROCm)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                gpu_info = {
                    "has_gpu": True,
                    "type": "rocm",
                    "name": "AMD GPU (ROCm)",
                }
                return gpu_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try Apple Silicon (Metal Performance Shaders)
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                cpu_brand = result.stdout.strip().lower()
                if "apple" in cpu_brand or "m1" in cpu_brand or "m2" in cpu_brand or "m3" in cpu_brand or "m4" in cpu_brand:
                    gpu_info = {
                        "has_gpu": True,
                        "type": "mps",  # Metal Performance Shaders
                        "name": f"Apple Silicon ({result.stdout.strip()})",
                    }
                    return gpu_info
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # No GPU found
        gpu_info["type"] = "cpu"
        return gpu_info

    def _calculate_tier(
        self, cpu_count: int, cpu_freq_mhz: float, ram_gb: float, has_gpu: bool
    ) -> str:
        """Calculate performance tier based on hardware"""
        score = 0

        # CPU score (0-30 points)
        if cpu_count >= 8:
            score += 15
        elif cpu_count >= 4:
            score += 10
        else:
            score += 5

        if cpu_freq_mhz >= 3000:
            score += 15
        elif cpu_freq_mhz >= 2000:
            score += 10
        else:
            score += 5

        # RAM score (0-30 points)
        if ram_gb >= 32:
            score += 30
        elif ram_gb >= 16:
            score += 20
        elif ram_gb >= 8:
            score += 10
        else:
            score += 5

        # GPU score (0-40 points)
        if has_gpu:
            score += 40

        # Determine tier
        if score >= 80:
            return "high"      # High-end workstation/gaming PC
        elif score >= 50:
            return "medium"    # Mid-range laptop/desktop
        else:
            return "low"       # Budget laptop/older hardware

    def check_ollama_models(self) -> List[str]:
        """Check which Ollama models are available"""
        print("🔍 Checking Ollama models...")
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                models = [m["name"] for m in data.get("models", [])]
                self.ollama_models = models
                return models
        except Exception as e:
            print(f"⚠️  Could not check Ollama models: {e}")
        return []

    def benchmark_model(self, model: str) -> Optional[float]:
        """Benchmark a specific model's response time"""
        print(f"⏱️  Benchmarking {model}...")
        try:
            start = time.time()
            result = subprocess.run(
                [
                    "curl", "-s", "-X", "POST",
                    "http://localhost:11434/api/generate",
                    "-d", json.dumps({
                        "model": model,
                        "prompt": "Hello",
                        "stream": False,
                    })
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            latency = (time.time() - start) * 1000  # ms

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    if "response" in data:
                        print(f"   ✅ {model}: {latency:.0f}ms")
                        return latency
                except json.JSONDecodeError:
                    pass

            print(f"   ❌ {model}: Failed")
            return None
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {model}: Timeout (>60s)")
            return None
        except Exception as e:
            print(f"   ❌ {model}: {e}")
            return None

    def recommend_config(self) -> RecommendedConfig:
        """Generate recommended configuration based on system profile"""
        if not self.hardware:
            self.profile_system()

        hw = self.hardware

        # Check available models
        available_models = self.check_ollama_models()

        # Benchmark available models if possible
        model_benchmarks = {}
        fast_models = ["qwen2:1.5b", "phi3:mini", "gemma:2b", "llama3.2:latest"]
        for model in fast_models:
            if model in available_models:
                latency = self.benchmark_model(model)
                if latency:
                    model_benchmarks[model] = latency

        # Select best model based on benchmarks
        if model_benchmarks:
            best_model = min(model_benchmarks, key=model_benchmarks.get)
            avg_latency = model_benchmarks[best_model]
        else:
            # Fallback to phi3:mini if no benchmarks
            best_model = "phi3:mini"
            avg_latency = 10000  # Assume 10s default

        # HIGH TIER: High-end workstation/gaming PC
        if hw.tier == "high":
            return RecommendedConfig(
                slm_timeout=60,
                slm_max_tokens=512,
                slm_model=best_model,
                dqn_batch_size=64,
                dqn_train_every=4,
                rag_n_results=5,
                rag_max_context_tokens=500,
                profile_name="High Performance",
                reasoning=f"""
🚀 HIGH PERFORMANCE Profile
- CPU: {hw.cpu_count} cores @ {hw.cpu_freq_mhz:.0f}MHz
- RAM: {hw.total_ram_gb:.1f}GB
- GPU: {hw.gpu_name or 'None'}
- Tier: {hw.tier.upper()}

Your system can handle intensive AI workloads.
- Model: {best_model} (avg latency: {avg_latency:.0f}ms)
- Full DQN training (batch 64, train every 4 steps)
- Rich RAG context (5 docs, 500 tokens)
- Generous timeout (60s) for complex queries
                """.strip()
            )

        # MEDIUM TIER: Mid-range laptop/desktop
        elif hw.tier == "medium":
            return RecommendedConfig(
                slm_timeout=45,
                slm_max_tokens=384,
                slm_model=best_model,
                dqn_batch_size=32,
                dqn_train_every=8,
                rag_n_results=4,
                rag_max_context_tokens=400,
                profile_name="Balanced",
                reasoning=f"""
⚖️  BALANCED Profile
- CPU: {hw.cpu_count} cores @ {hw.cpu_freq_mhz:.0f}MHz
- RAM: {hw.total_ram_gb:.1f}GB
- GPU: {hw.gpu_name or 'None'}
- Tier: {hw.tier.upper()}

Optimized for good performance with reasonable resource usage.
- Model: {best_model} (avg latency: {avg_latency:.0f}ms)
- Moderate DQN training (batch 32, train every 8 steps)
- Good RAG context (4 docs, 400 tokens)
- Balanced timeout (45s)
                """.strip()
            )

        # LOW TIER: Budget laptop/older hardware
        else:  # low tier
            return RecommendedConfig(
                slm_timeout=30,
                slm_max_tokens=256,
                slm_model=best_model,
                dqn_batch_size=16,
                dqn_train_every=12,
                rag_n_results=3,
                rag_max_context_tokens=300,
                profile_name="Efficient",
                reasoning=f"""
💡 EFFICIENT Profile
- CPU: {hw.cpu_count} cores @ {hw.cpu_freq_mhz:.0f}MHz
- RAM: {hw.total_ram_gb:.1f}GB
- GPU: {hw.gpu_name or 'None'}
- Tier: {hw.tier.upper()}

Optimized for resource-constrained systems.
- Model: {best_model} (avg latency: {avg_latency:.0f}ms)
- Light DQN training (batch 16, train every 12 steps)
- Minimal RAG context (3 docs, 300 tokens)
- Fast timeout (30s) for quick responses
                """.strip()
            )

    def print_profile(self):
        """Pretty-print system profile"""
        if not self.hardware:
            self.profile_system()

        hw = self.hardware

        print("\n" + "="*60)
        print("🖥️  SYSTEM PROFILE")
        print("="*60)

        print(f"\n🔧 CPU:")
        print(f"   • Brand: {hw.cpu_brand}")
        print(f"   • Cores: {hw.cpu_count}")
        print(f"   • Frequency: {hw.cpu_freq_mhz:.0f} MHz")

        print(f"\n💾 Memory:")
        print(f"   • Total: {hw.total_ram_gb:.1f} GB")
        print(f"   • Available: {hw.available_ram_gb:.1f} GB")

        print(f"\n🎮 GPU:")
        if hw.has_gpu:
            print(f"   • Type: {hw.gpu_type.upper()}")
            print(f"   • Name: {hw.gpu_name}")
            if hw.gpu_vram_gb:
                print(f"   • VRAM: {hw.gpu_vram_gb:.1f} GB")
        else:
            print(f"   • No GPU detected (CPU only)")

        print(f"\n🖥️  Operating System:")
        print(f"   • OS: {hw.os_name}")
        print(f"   • Version: {hw.os_version}")

        print(f"\n⚡ Performance Tier: {hw.tier.upper()}")

        if self.ollama_models:
            print(f"\n🤖 Ollama Models Available:")
            for model in self.ollama_models[:5]:  # Show first 5
                print(f"   • {model}")
            if len(self.ollama_models) > 5:
                print(f"   ... and {len(self.ollama_models) - 5} more")

        print("\n" + "="*60 + "\n")

    def print_recommendation(self, config: RecommendedConfig):
        """Pretty-print recommended configuration"""
        print("\n" + "="*60)
        print(f"✨ RECOMMENDED CONFIGURATION: {config.profile_name}")
        print("="*60)

        print(config.reasoning)

        print(f"\n📝 Suggested Settings:")
        print(f"\nSLM:")
        print(f"   • Model: {config.slm_model}")
        print(f"   • Timeout: {config.slm_timeout}s")
        print(f"   • Max Tokens: {config.slm_max_tokens}")

        print(f"\nDQN:")
        print(f"   • Batch Size: {config.dqn_batch_size}")
        print(f"   • Train Every: {config.dqn_train_every} steps")

        print(f"\nRAG:")
        print(f"   • Results: {config.rag_n_results} documents")
        print(f"   • Max Context: {config.rag_max_context_tokens} tokens")

        print("\n" + "="*60 + "\n")

    def write_config(self, config: RecommendedConfig, output_path: str):
        """Write recommended config to YAML file"""
        from pathlib import Path

        # Read current config
        with open(output_path, "r") as f:
            lines = f.readlines()

        # Update specific lines
        new_lines = []
        for line in lines:
            if line.strip().startswith("timeout:"):
                new_lines.append(f"  timeout: {config.slm_timeout}                        # ⚡ Auto-configured based on hardware\n")
            elif line.strip().startswith("max_tokens:"):
                new_lines.append(f"  max_tokens: {config.slm_max_tokens}                    # ⚡ Auto-configured based on hardware\n")
            elif line.strip().startswith("model:") and "phi3" in line:
                new_lines.append(f'  model: "{config.slm_model}"\n')
            elif line.strip().startswith("batch_size:"):
                new_lines.append(f"  batch_size: {config.dqn_batch_size}                     # ⚡ Auto-configured based on hardware\n")
            elif line.strip().startswith("train_every:"):
                new_lines.append(f"  train_every: {config.dqn_train_every}                    # ⚡ Auto-configured based on hardware\n")
            elif line.strip().startswith("default_n_results:"):
                new_lines.append(f"  default_n_results: {config.rag_n_results}               # ⚡ Auto-configured based on hardware\n")
            elif line.strip().startswith("max_context_tokens:"):
                new_lines.append(f"  max_context_tokens: {config.rag_max_context_tokens}            # ⚡ Auto-configured based on hardware\n")
            else:
                new_lines.append(line)

        # Write updated config
        with open(output_path, "w") as f:
            f.writelines(new_lines)

        print(f"✅ Configuration written to: {output_path}")


def main():
    """CLI interface for system profiler"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dheera System Profiler - Detect hardware and recommend optimal configuration"
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Only show system profile without recommendations",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write recommended config to dheera_config.yaml",
    )
    parser.add_argument(
        "--config-path",
        default="config/dheera_config.yaml",
        help="Path to config file (default: config/dheera_config.yaml)",
    )

    args = parser.parse_args()

    profiler = SystemProfiler()

    # Profile system
    profiler.profile_system()
    profiler.print_profile()

    if args.profile_only:
        return

    # Get recommendations
    config = profiler.recommend_config()
    profiler.print_recommendation(config)

    # Write config if requested
    if args.write_config:
        profiler.write_config(config, args.config_path)
        print("\n✅ Configuration updated! Restart Dheera to apply changes.")
    else:
        print("\n💡 To apply these settings, run:")
        print(f"   python3 utils/system_profiler.py --write-config")


if __name__ == "__main__":
    main()
