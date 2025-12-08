"""
InferaGrid Resource Categories
Defines all possible categories for compute resources
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class InferenceSpeed(Enum):
    """Inference speed tier based on hardware"""
    ULTRA_FAST = "ultra_fast"      # Datacenter GPU (A100, H100)
    FAST = "fast"                   # High-end consumer GPU (RTX 4090, 4080)
    STANDARD = "standard"           # Mid-tier GPU (RTX 3070, 3060)
    MODERATE = "moderate"           # Entry GPU or Apple Silicon
    SLOW = "slow"                   # Integrated GPU
    BATCH_ONLY = "batch_only"       # CPU-only, not suitable for real-time


class ComputeTier(Enum):
    """Overall compute power tier"""
    DATACENTER = "datacenter"       # Enterprise-grade hardware
    PROFESSIONAL = "professional"   # Workstation-grade
    PROSUMER = "prosumer"           # High-end consumer
    CONSUMER = "consumer"           # Standard consumer
    LIGHT = "light"                 # Basic/entry level


class GPUCapability(Enum):
    """GPU capability classification"""
    LLM_70B_PLUS = "llm_70b_plus"           # Can run 70B+ models (48GB+ VRAM)
    LLM_30B = "llm_30b"                      # Can run 30B models (24GB+ VRAM)
    LLM_13B = "llm_13b"                      # Can run 13B models (16GB+ VRAM)
    LLM_7B = "llm_7b"                        # Can run 7B models (8GB+ VRAM)
    LLM_3B = "llm_3b"                        # Can run 3B models (4GB+ VRAM)
    IMAGE_GEN_HD = "image_gen_hd"            # HD image generation (12GB+ VRAM)
    IMAGE_GEN_SD = "image_gen_sd"            # Standard image generation (8GB+ VRAM)
    LIGHT_INFERENCE = "light_inference"      # Light inference tasks
    CPU_ONLY = "cpu_only"                    # No GPU acceleration


class WorkloadType(Enum):
    """Suitable workload types"""
    REAL_TIME_INFERENCE = "real_time_inference"
    BATCH_INFERENCE = "batch_inference"
    MODEL_TRAINING = "model_training"
    FINE_TUNING = "fine_tuning"
    DATA_PROCESSING = "data_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_PROCESSING = "video_processing"
    GENERAL_COMPUTE = "general_compute"


class AvailabilityTier(Enum):
    """Expected availability/reliability tier"""
    ALWAYS_ON = "always_on"         # 99%+ uptime expected
    BUSINESS_HOURS = "business_hours"  # Available during work hours
    VARIABLE = "variable"           # Availability varies
    SPOT = "spot"                   # Can be preempted anytime


@dataclass
class CategoryDefinition:
    """Full category definition for a compute node"""
    inference_speed: InferenceSpeed
    compute_tier: ComputeTier
    gpu_capability: GPUCapability
    supported_workloads: List[WorkloadType]
    availability_tier: AvailabilityTier
    
    # Pricing multipliers
    base_rate_multiplier: float
    demand_factor: float
    
    # Human-readable
    display_name: str
    description: str
    recommended_tasks: List[str]
    
    # Limits
    max_concurrent_jobs: int
    max_context_length: int  # For LLM inference
    max_batch_size: int


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY CLASSIFICATION RULES
# ═══════════════════════════════════════════════════════════════════════════════

# GPU Name patterns for classification
DATACENTER_GPUS = ["A100", "H100", "H200", "A6000", "RTX 6000", "L40", "A40"]
PROFESSIONAL_GPUS = ["RTX 4090", "RTX 4080", "RTX 3090", "TITAN", "A5000", "RTX 5000"]
PROSUMER_GPUS = ["RTX 4070", "RTX 3080", "RTX 3070", "RX 7900", "RX 6900"]
CONSUMER_GPUS = ["RTX 4060", "RTX 3060", "RTX 2080", "RTX 2070", "RX 7800", "RX 6800"]
ENTRY_GPUS = ["RTX 2060", "GTX 1660", "GTX 1650", "RX 6600", "RX 5600"]
APPLE_SILICON = ["M1", "M2", "M3", "M4"]
INTEGRATED_GPUS = ["INTEL", "UHD", "IRIS", "RADEON GRAPHICS"]


def classify_gpu_tier(gpu_name: str) -> tuple:
    """Classify GPU into tier and capability"""
    gpu_upper = gpu_name.upper()
    
    # Check datacenter GPUs
    for pattern in DATACENTER_GPUS:
        if pattern in gpu_upper:
            return ComputeTier.DATACENTER, InferenceSpeed.ULTRA_FAST
    
    # Check professional GPUs
    for pattern in PROFESSIONAL_GPUS:
        if pattern in gpu_upper:
            return ComputeTier.PROFESSIONAL, InferenceSpeed.FAST
    
    # Check prosumer GPUs
    for pattern in PROSUMER_GPUS:
        if pattern in gpu_upper:
            return ComputeTier.PROSUMER, InferenceSpeed.STANDARD
    
    # Check consumer GPUs
    for pattern in CONSUMER_GPUS:
        if pattern in gpu_upper:
            return ComputeTier.CONSUMER, InferenceSpeed.STANDARD
    
    # Check entry GPUs
    for pattern in ENTRY_GPUS:
        if pattern in gpu_upper:
            return ComputeTier.CONSUMER, InferenceSpeed.MODERATE
    
    # Check Apple Silicon
    for pattern in APPLE_SILICON:
        if pattern in gpu_upper:
            return ComputeTier.PROSUMER, InferenceSpeed.MODERATE
    
    # Check integrated GPUs
    for pattern in INTEGRATED_GPUS:
        if pattern in gpu_upper:
            return ComputeTier.LIGHT, InferenceSpeed.SLOW
    
    # Default
    return ComputeTier.LIGHT, InferenceSpeed.BATCH_ONLY


def classify_gpu_capability(vram_mb: Optional[int], gpu_name: str) -> GPUCapability:
    """Classify GPU capability based on VRAM"""
    if not vram_mb:
        # Try to estimate from GPU name
        gpu_upper = gpu_name.upper()
        if any(p in gpu_upper for p in DATACENTER_GPUS):
            vram_mb = 48000  # Assume high VRAM
        elif any(p in gpu_upper for p in PROFESSIONAL_GPUS):
            vram_mb = 24000
        elif any(p in gpu_upper for p in INTEGRATED_GPUS):
            vram_mb = 1536
        else:
            vram_mb = 0
    
    if vram_mb >= 48000:
        return GPUCapability.LLM_70B_PLUS
    elif vram_mb >= 24000:
        return GPUCapability.LLM_30B
    elif vram_mb >= 16000:
        return GPUCapability.LLM_13B
    elif vram_mb >= 8000:
        return GPUCapability.LLM_7B
    elif vram_mb >= 4000:
        return GPUCapability.LLM_3B
    elif vram_mb >= 2000:
        return GPUCapability.LIGHT_INFERENCE
    else:
        return GPUCapability.CPU_ONLY


def get_supported_workloads(
    gpu_capability: GPUCapability,
    cpu_cores: float,
    memory_gb: float
) -> List[WorkloadType]:
    """Determine supported workloads based on resources"""
    workloads = [WorkloadType.GENERAL_COMPUTE]
    
    # GPU-based workloads
    if gpu_capability != GPUCapability.CPU_ONLY:
        workloads.append(WorkloadType.BATCH_INFERENCE)
        
        if gpu_capability in [GPUCapability.LLM_70B_PLUS, GPUCapability.LLM_30B, 
                              GPUCapability.LLM_13B, GPUCapability.LLM_7B]:
            workloads.append(WorkloadType.REAL_TIME_INFERENCE)
            workloads.append(WorkloadType.EMBEDDING_GENERATION)
        
        if gpu_capability in [GPUCapability.LLM_70B_PLUS, GPUCapability.LLM_30B]:
            workloads.append(WorkloadType.MODEL_TRAINING)
            workloads.append(WorkloadType.FINE_TUNING)
        
        if gpu_capability in [GPUCapability.LLM_70B_PLUS, GPUCapability.LLM_30B,
                              GPUCapability.LLM_13B, GPUCapability.IMAGE_GEN_HD,
                              GPUCapability.IMAGE_GEN_SD]:
            workloads.append(WorkloadType.IMAGE_GENERATION)
    
    # CPU-based workloads
    if cpu_cores >= 4 and memory_gb >= 8:
        workloads.append(WorkloadType.DATA_PROCESSING)
    
    if cpu_cores >= 8 and memory_gb >= 16:
        workloads.append(WorkloadType.VIDEO_PROCESSING)
    
    return workloads


def get_pricing_multiplier(compute_tier: ComputeTier, gpu_capability: GPUCapability) -> float:
    """Get pricing multiplier based on tier"""
    tier_multipliers = {
        ComputeTier.DATACENTER: 10.0,
        ComputeTier.PROFESSIONAL: 5.0,
        ComputeTier.PROSUMER: 2.5,
        ComputeTier.CONSUMER: 1.5,
        ComputeTier.LIGHT: 1.0,
    }
    
    capability_bonus = {
        GPUCapability.LLM_70B_PLUS: 2.0,
        GPUCapability.LLM_30B: 1.5,
        GPUCapability.LLM_13B: 1.2,
        GPUCapability.LLM_7B: 1.1,
        GPUCapability.LLM_3B: 1.0,
        GPUCapability.IMAGE_GEN_HD: 1.3,
        GPUCapability.IMAGE_GEN_SD: 1.1,
        GPUCapability.LIGHT_INFERENCE: 0.8,
        GPUCapability.CPU_ONLY: 0.5,
    }
    
    return tier_multipliers.get(compute_tier, 1.0) * capability_bonus.get(gpu_capability, 1.0)


def classify_node(
    cpu_cores: float,
    memory_gb: float,
    storage_gb: float,
    gpu_name: Optional[str],
    vram_mb: Optional[int]
) -> CategoryDefinition:
    """
    Classify a compute node into appropriate categories
    Returns full CategoryDefinition
    """
    
    # Determine GPU characteristics
    if gpu_name:
        compute_tier, inference_speed = classify_gpu_tier(gpu_name)
        gpu_capability = classify_gpu_capability(vram_mb, gpu_name)
    else:
        compute_tier = ComputeTier.LIGHT
        inference_speed = InferenceSpeed.BATCH_ONLY
        gpu_capability = GPUCapability.CPU_ONLY
    
    # Adjust tier based on CPU/Memory
    if cpu_cores >= 16 and memory_gb >= 64:
        if compute_tier == ComputeTier.LIGHT:
            compute_tier = ComputeTier.CONSUMER
    elif cpu_cores >= 8 and memory_gb >= 32:
        if compute_tier == ComputeTier.LIGHT:
            compute_tier = ComputeTier.CONSUMER
    
    # Get supported workloads
    workloads = get_supported_workloads(gpu_capability, cpu_cores, memory_gb)
    
    # Get pricing
    pricing_multiplier = get_pricing_multiplier(compute_tier, gpu_capability)
    
    # Generate display name and description
    display_name = f"{compute_tier.value.title()} - {inference_speed.value.replace('_', ' ').title()}"
    
    if gpu_name:
        description = f"{gpu_name} with {cpu_cores:.1f} CPU cores and {memory_gb:.1f}GB RAM"
    else:
        description = f"CPU-only node with {cpu_cores:.1f} cores and {memory_gb:.1f}GB RAM"
    
    # Recommended tasks
    recommended_tasks = []
    if WorkloadType.REAL_TIME_INFERENCE in workloads:
        recommended_tasks.append("Real-time LLM inference")
    if WorkloadType.IMAGE_GENERATION in workloads:
        recommended_tasks.append("AI image generation")
    if WorkloadType.EMBEDDING_GENERATION in workloads:
        recommended_tasks.append("Text embeddings")
    if WorkloadType.DATA_PROCESSING in workloads:
        recommended_tasks.append("Data processing pipelines")
    if WorkloadType.BATCH_INFERENCE in workloads:
        recommended_tasks.append("Batch inference jobs")
    if not recommended_tasks:
        recommended_tasks.append("Light compute tasks")
    
    # Calculate limits
    max_concurrent = max(1, int(cpu_cores / 2))
    max_context = 2048  # Default
    if gpu_capability == GPUCapability.LLM_70B_PLUS:
        max_context = 128000
    elif gpu_capability == GPUCapability.LLM_30B:
        max_context = 32000
    elif gpu_capability == GPUCapability.LLM_13B:
        max_context = 16000
    elif gpu_capability == GPUCapability.LLM_7B:
        max_context = 8000
    
    max_batch = max(1, int(memory_gb / 4))
    
    return CategoryDefinition(
        inference_speed=inference_speed,
        compute_tier=compute_tier,
        gpu_capability=gpu_capability,
        supported_workloads=workloads,
        availability_tier=AvailabilityTier.VARIABLE,  # Default for consumer nodes
        base_rate_multiplier=pricing_multiplier,
        demand_factor=1.0,
        display_name=display_name,
        description=description,
        recommended_tasks=recommended_tasks,
        max_concurrent_jobs=max_concurrent,
        max_context_length=max_context,
        max_batch_size=max_batch,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

ALL_CATEGORIES_INFO = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    INFERAGRID COMPUTE CATEGORIES                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─ INFERENCE SPEED TIERS ──────────────────────────────────────────────────────────┐
│                                                                                   │
│  ⚡ ULTRA_FAST    │ Datacenter GPUs (A100, H100)                                 │
│                   │ < 50ms latency, 100+ tokens/sec                              │
│                   │ Best for: Production APIs, real-time applications            │
│                                                                                   │
│  🚀 FAST          │ High-end GPUs (RTX 4090, 4080, 3090)                         │
│                   │ 50-150ms latency, 50-100 tokens/sec                          │
│                   │ Best for: Interactive apps, chatbots                         │
│                                                                                   │
│  ✨ STANDARD      │ Mid-tier GPUs (RTX 3070, 3060, 4060)                         │
│                   │ 150-500ms latency, 20-50 tokens/sec                          │
│                   │ Best for: Async inference, background tasks                  │
│                                                                                   │
│  ⏳ MODERATE      │ Entry GPUs, Apple Silicon (M1/M2/M3)                         │
│                   │ 500ms-2s latency, 10-30 tokens/sec                           │
│                   │ Best for: Development, testing, small models                 │
│                                                                                   │
│  🐢 SLOW          │ Integrated GPUs (Intel UHD, Iris)                            │
│                   │ 2-10s latency, 5-15 tokens/sec                               │
│                   │ Best for: Tiny models, embeddings only                       │
│                                                                                   │
│  📦 BATCH_ONLY    │ CPU-only nodes                                               │
│                   │ Variable latency, best for non-realtime                      │
│                   │ Best for: Data processing, batch jobs                        │
└───────────────────────────────────────────────────────────────────────────────────┘

┌─ COMPUTE TIERS ──────────────────────────────────────────────────────────────────┐
│                                                                                   │
│  🏢 DATACENTER     │ Enterprise hardware, 99.9% SLA potential                    │
│                    │ A100/H100, 64+ cores, 256GB+ RAM                            │
│                    │ Rate: $2.00 - $5.00 / hour                                  │
│                                                                                   │
│  💼 PROFESSIONAL   │ Workstation-grade, high reliability                         │
│                    │ RTX 4090/3090, 16+ cores, 64GB+ RAM                         │
│                    │ Rate: $0.80 - $2.00 / hour                                  │
│                                                                                   │
│  🎮 PROSUMER       │ High-end consumer, good performance                         │
│                    │ RTX 3080/4070, 8+ cores, 32GB+ RAM                          │
│                    │ Rate: $0.30 - $0.80 / hour                                  │
│                                                                                   │
│  🖥️ CONSUMER       │ Standard consumer hardware                                  │
│                    │ RTX 3060/4060, 4+ cores, 16GB+ RAM                          │
│                    │ Rate: $0.10 - $0.30 / hour                                  │
│                                                                                   │
│  💡 LIGHT          │ Basic/entry-level hardware                                  │
│                    │ Integrated GPU or CPU-only                                  │
│                    │ Rate: $0.01 - $0.10 / hour                                  │
└───────────────────────────────────────────────────────────────────────────────────┘

┌─ GPU CAPABILITY LEVELS ──────────────────────────────────────────────────────────┐
│                                                                                   │
│  🦙 LLM_70B_PLUS   │ 48GB+ VRAM - Can run Llama 70B, Mixtral 8x22B              │
│  🦙 LLM_30B        │ 24GB+ VRAM - Can run Llama 30B, CodeLlama 34B              │
│  🦙 LLM_13B        │ 16GB+ VRAM - Can run Llama 13B, Mistral 7B (high context)  │
│  🦙 LLM_7B         │ 8GB+ VRAM  - Can run Llama 7B, Phi-3, Gemma 7B             │
│  🦙 LLM_3B         │ 4GB+ VRAM  - Can run Phi-2, TinyLlama, Gemma 2B            │
│  🎨 IMAGE_GEN_HD   │ 12GB+ VRAM - HD Stable Diffusion, SDXL                      │
│  🎨 IMAGE_GEN_SD   │ 8GB+ VRAM  - Standard Stable Diffusion                      │
│  ⚙️ LIGHT_INFERENCE│ 2-4GB VRAM - Small models, embeddings                       │
│  🔲 CPU_ONLY       │ No GPU     - CPU inference only                             │
└───────────────────────────────────────────────────────────────────────────────────┘

┌─ SUPPORTED WORKLOAD TYPES ───────────────────────────────────────────────────────┐
│                                                                                   │
│  • REAL_TIME_INFERENCE   - Interactive LLM chat, <500ms response                 │
│  • BATCH_INFERENCE       - Bulk processing, latency not critical                 │
│  • MODEL_TRAINING        - Full model training (requires high VRAM)              │
│  • FINE_TUNING           - LoRA/QLoRA fine-tuning                                │
│  • DATA_PROCESSING       - ETL, data transformation pipelines                    │
│  • EMBEDDING_GENERATION  - Vector embeddings for RAG/search                      │
│  • IMAGE_GENERATION      - Stable Diffusion, DALL-E style generation            │
│  • VIDEO_PROCESSING      - Video transcoding, analysis                           │
│  • GENERAL_COMPUTE       - Generic compute tasks                                 │
└───────────────────────────────────────────────────────────────────────────────────┘
"""


def print_all_categories():
    """Print all category information"""
    print(ALL_CATEGORIES_INFO)


if __name__ == "__main__":
    print_all_categories()
