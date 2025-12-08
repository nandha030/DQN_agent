# InferaGrid Provider Agent

A complete toolkit to share your compute resources with the InferaGrid decentralized marketplace.

## 🗂️ Files Included

| File | Description |
|------|-------------|
| `system_profiler.py` | Detects CPU, GPU, Memory, Storage on your machine |
| `resource_locker.py` | Interactive tool to lock resources with 20% safety buffer |
| `categories.py` | Node classification system (speed tiers, compute tiers, etc.) |
| `api_server.py` | FastAPI server that exposes your node to InferaGrid Exchange |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Profile Your System
```bash
python system_profiler.py
```

### 3. Lock Resources (with 20% safety buffer)
```bash
python resource_locker.py
```

### 4. Start API Server
```bash
python api_server.py
```

Your node is now available at http://localhost:8420

## 📊 Categories Reference

Run: `python resource_locker.py --categories`

### Inference Speed Tiers
- ⚡ ULTRA_FAST - A100/H100 (<50ms)
- 🚀 FAST - RTX 4090/4080 (50-150ms)
- ✨ STANDARD - RTX 3070/3060 (150-500ms)
- ⏳ MODERATE - Apple Silicon (500ms-2s)
- 🐢 SLOW - Intel UHD/Iris (2-10s)
- 📦 BATCH_ONLY - CPU-only

### Compute Tiers
- 🏢 DATACENTER ($2-5/hr)
- 💼 PROFESSIONAL ($0.80-2/hr)
- 🎮 PROSUMER ($0.30-0.80/hr)
- 🖥️ CONSUMER ($0.10-0.30/hr)
- 💡 LIGHT ($0.01-0.10/hr)

### GPU Capability
- LLM_70B_PLUS (48GB+ VRAM)
- LLM_30B (24GB+ VRAM)
- LLM_13B (16GB+ VRAM)
- LLM_7B (8GB+ VRAM)
- CPU_ONLY (No GPU)

## 🔌 API Endpoints

| Endpoint | Description |
|----------|-------------|
| GET /health | Health check |
| GET /node/info | Full node info |
| GET /node/metrics | Real-time metrics |
| POST /jobs/submit | Submit a job |
| GET /jobs/{id} | Job status |

## 🛠️ CLI Commands
```bash
python system_profiler.py --json          # Export as JSON
python resource_locker.py --categories    # Show all categories
python resource_locker.py --status        # Check lock status
python resource_locker.py --unlock        # Remove lock
python api_server.py --port 9000          # Custom port
```
