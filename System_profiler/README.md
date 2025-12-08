# InferaGrid System Profiler

Detects your system resources (CPU, GPU, Memory, Storage) and shows what you can offer to the InferaGrid marketplace.

## Quick Start (macOS)

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install psutil

# 3. Run the profiler
python system_profiler.py
```

## Output Formats

### Pretty Terminal Output (default)
```bash
python system_profiler.py
```

### JSON Export (for API integration)
```bash
python system_profiler.py --json
```

### Save JSON to file
```bash
python system_profiler.py --json > my_system.json
```

## What It Detects

| Resource | Detection Method |
|----------|------------------|
| CPU | `psutil` + `sysctl` (macOS) |
| Memory | `psutil` |
| GPU (NVIDIA) | `GPUtil` library |
| GPU (macOS) | `system_profiler` command |
| GPU (Apple Silicon) | Detects M1/M2/M3 chips |
| Disk | `psutil` |
| Network | `psutil` |

## Marketplace Offering Calculation

The profiler calculates what you can safely offer:
- **70%** of available CPU cores
- **70%** of available RAM
- **50%** of available storage (max 100GB)
- **GPU**: Full availability if detected

## Pricing Tiers

| Tier | Example Hardware | Est. Rate |
|------|------------------|-----------|
| CPU | Intel i5/i7, M1/M2 | $0.01/core/hr |
| GPU_CONSUMER | RTX 3060, GTX 1080 | $0.30/hr |
| GPU_PROFESSIONAL | RTX 4090, RTX 3090 | $0.80/hr |
| GPU_DATACENTER | A100, H100 | $2.50/hr |

## Next Steps

This profiler is the first component of the InferaGrid Provider Agent. Coming next:
1. **Provider Agent** - Register with InferaGrid, send heartbeats
2. **Job Handler** - Accept and execute compute jobs
3. **API Server** - Central orchestration service
