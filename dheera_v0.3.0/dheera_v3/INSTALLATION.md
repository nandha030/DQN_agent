# 📦 Dheera v0.3.0 - Complete Installation Guide

This guide covers the complete installation of Dheera including all databases, backend, and frontend components.

---

## 🎯 Quick Installation (Automated)

### One-Command Setup
```bash
chmod +x setup.sh
./setup.sh
```

This automated script installs:
- ✅ Python virtual environment
- ✅ Ollama (local LLM runtime)
- ✅ Milvus (vector database)
- ✅ Neo4j (knowledge graph)
- ✅ Redis (cache)
- ✅ PostgreSQL + TimescaleDB
- ✅ All Python dependencies
- ✅ Database initialization
- ✅ Configuration files

**Time:** 15-30 minutes (depending on internet speed)

**After installation:**
```bash
./start_all.sh              # Start all services
open http://localhost:8501  # Open web GUI
```

---

## 📋 Prerequisites

### System Requirements

**Minimum:**
- OS: macOS 10.15+ or Ubuntu 20.04+
- RAM: 8GB
- Disk: 20GB free space
- CPU: 4 cores
- Internet connection

**Recommended:**
- OS: macOS 13+ or Ubuntu 22.04+
- RAM: 16GB+
- Disk: 50GB+ free space
- CPU: 8+ cores
- GPU: NVIDIA with CUDA 12.1+ (optional)

### Software Requirements

**Must be installed before running setup.sh:**
1. **Python 3.11+**
   ```bash
   python3 --version  # Should show 3.11 or higher
   ```

2. **Docker Desktop** (macOS) or **Docker + Docker Compose** (Linux)
   ```bash
   docker --version
   docker-compose --version
   ```

3. **Git** (usually pre-installed)
   ```bash
   git --version
   ```

---

## 🚀 Step-by-Step Installation

### Option A: Automated (Recommended)

```bash
# 1. Clone repository (if not already done)
cd /Users/nandhavignesh/triton/dheera_v0.3.0/dheera_v3

# 2. Make setup script executable
chmod +x setup.sh

# 3. Run automated setup
./setup.sh

# 4. Wait for completion (15-30 minutes)
# Script will:
# - Check system requirements
# - Install Ollama and pull models
# - Setup Python virtual environment
# - Install all dependencies
# - Start Docker containers for databases
# - Initialize all databases
# - Create configuration files
# - Create startup scripts

# 5. Verify installation
./verify_installation.sh

# 6. Start all services
./start_all.sh
```

### Option B: Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

#### 1. Install Ollama
```bash
# macOS or Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve &

# Pull models (takes 10-20 minutes)
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

#### 2. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_full.txt
```

#### 3. Setup Databases with Docker
```bash
# Start all database services
docker-compose up -d

# Wait for services to be ready
sleep 30

# Check status
docker-compose ps
```

#### 4. Initialize Databases
```bash
# Run initialization script
python3 scripts/init_databases.py
```

#### 5. Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit if needed
nano .env
```

#### 6. Start Services
```bash
# Start backend
./start_backend.sh &

# Start frontend
./start_frontend.sh &
```

</details>

---

## 🗂️ What Gets Installed

### Directory Structure
```
dheera_v3/
├── setup.sh                    # Main installation script
├── docker-compose.yml          # Database services
├── .env                        # Environment configuration
├── requirements.txt            # Core dependencies
├── requirements_full.txt       # All dependencies
│
├── venv/                       # Python virtual environment
├── data/                       # Database data
│   ├── milvus/                 # Vector database
│   ├── neo4j/                  # Knowledge graph
│   ├── redis/                  # Cache
│   └── postgresql/             # Relational database
│
├── logs/                       # Application logs
├── backups/                    # Database backups
├── checkpoints/                # Model checkpoints
├── uploads/                    # Uploaded documents
│
├── scripts/
│   └── init_databases.py       # Database initialization
│
├── start_all.sh                # Start all services
├── start_backend.sh            # Start backend only
├── start_frontend.sh           # Start frontend only
├── start_chat.sh               # Start CLI chat
├── stop_all.sh                 # Stop all services
│
└── verify_installation.sh      # Verify installation
```

### Docker Containers

| Container | Image | Port(s) | Purpose |
|-----------|-------|---------|---------|
| milvus-standalone | milvusdb/milvus:v2.3.3 | 19530, 9091 | Vector database |
| milvus-etcd | quay.io/coreos/etcd:v3.5.5 | 2379 | Milvus metadata |
| milvus-minio | minio/minio | 9000, 9001 | Milvus storage |
| dheera-neo4j | neo4j:5.14.0 | 7474, 7687 | Knowledge graph |
| dheera-redis | redis:7.2-alpine | 6379 | Cache |
| dheera-postgresql | timescale/timescaledb | 5432 | Relational DB |
| dheera-adminer | adminer:latest | 8080 | DB web UI |

### Python Packages (Total: 50+)

**Core (10):**
- torch, numpy, scipy, pyyaml, requests, tqdm, rich, loguru, psutil, python-dotenv

**Databases (8):**
- pymilvus, chromadb, neo4j, py2neo, redis, hiredis, psycopg2-binary, sqlalchemy

**LLM & AI (7):**
- sentence-transformers, transformers, tokenizers, openai, anthropic

**API & Web (8):**
- fastapi, uvicorn, pydantic, websockets, httpx, aiohttp, streamlit

**Document Processing (7):**
- PyPDF2, python-docx, python-pptx, openpyxl, pandas, Pillow, pytesseract

**Visualization (6):**
- plotly, matplotlib, seaborn, networkx, pyvis

**Testing & Quality (6):**
- pytest, pytest-cov, pytest-asyncio, black, flake8, mypy

---

## 🔧 Configuration

### Environment Variables (.env)

Key configuration options:

```bash
# LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Database Connections
MILVUS_HOST=localhost
MILVUS_PORT=19530
NEO4J_URI=bolt://localhost:7687
REDIS_HOST=localhost
POSTGRES_HOST=localhost

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Settings
STREAMLIT_SERVER_PORT=8501

# Performance
SPIKING_ENABLED=true
USE_GPU=false
```

### Dheera Configuration (config/dheera_config.yaml)

Main settings:

```yaml
llm:
  provider: "ollama"
  model: "llama3.2:3b"
  temperature: 0.7
  max_tokens: 512

vector_db:
  provider: "milvus"
  host: "localhost"
  port: 19530

spiking:
  enabled: true
  time_steps: 5

rag:
  enabled: true
  chunk_size: 500
  default_n_results: 3
```

---

## ✅ Verification

### Check Installation Status

```bash
./verify_installation.sh
```

Expected output:
```
✅ Python installed
✅ Ollama installed
✅ Docker installed
✅ Docker Compose installed

📊 Checking Database Services...
✅ Milvus running
✅ Neo4j running
✅ Redis running
✅ PostgreSQL running

📦 Checking Python Packages...
✅ torch installed
✅ fastapi installed
✅ streamlit installed
✅ pymilvus installed
...

🎯 Checking Ollama Models...
✅ llama3.2:3b pulled
✅ nomic-embed-text pulled
```

### Test Services

```bash
# Test Milvus
python3 -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('✅ Milvus OK')"

# Test Neo4j
python3 -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'dheera123')); driver.verify_connectivity(); print('✅ Neo4j OK')"

# Test Redis
python3 -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping(); print('✅ Redis OK')"

# Test PostgreSQL
python3 -c "import psycopg2; conn = psycopg2.connect(host='localhost', port=5432, user='dheera', password='dheera123', database='dheera_db'); print('✅ PostgreSQL OK')"
```

### Test Dheera

```bash
# CLI test
./start_chat.sh

# Type: "Hello Dheera"
# Expected: Response within 1-3 seconds
```

---

## 🚀 Starting Services

### Start Everything
```bash
./start_all.sh
```

Access points:
- **Frontend GUI:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Neo4j Browser:** http://localhost:7474 (neo4j / dheera123)
- **Database Admin:** http://localhost:8080

### Start Individual Services

```bash
# Backend only
./start_backend.sh

# Frontend only
./start_frontend.sh

# CLI chat only
./start_chat.sh

# Databases only
docker-compose up -d
```

### Stop Services

```bash
# Stop all
./stop_all.sh

# Stop databases
docker-compose down

# Keep databases running, stop apps only
kill $(cat logs/backend.pid)
kill $(cat logs/frontend.pid)
```

---

## 🐛 Troubleshooting

### Setup Script Fails

**Problem:** `setup.sh` exits with error

**Solutions:**
1. Check system requirements:
   ```bash
   python3 --version  # Must be 3.11+
   docker --version   # Must be installed
   ```

2. Check disk space:
   ```bash
   df -h .  # Need 20GB+ free
   ```

3. Check internet connection:
   ```bash
   ping -c 3 ollama.com
   ```

4. Run in verbose mode:
   ```bash
   bash -x setup.sh 2>&1 | tee setup.log
   ```

### Docker Containers Won't Start

**Problem:** `docker-compose up -d` fails

**Solutions:**
1. Check Docker is running:
   ```bash
   docker ps
   ```

2. Check ports not in use:
   ```bash
   lsof -i :19530  # Milvus
   lsof -i :7474   # Neo4j
   lsof -i :6379   # Redis
   lsof -i :5432   # PostgreSQL
   ```

3. Increase Docker resources:
   - Docker Desktop → Settings → Resources
   - Set: 8GB RAM, 4 CPUs minimum

4. Clean and restart:
   ```bash
   docker-compose down -v
   docker system prune -a
   docker-compose up -d
   ```

### Ollama Models Won't Pull

**Problem:** `ollama pull llama3.2:3b` hangs or fails

**Solutions:**
1. Check Ollama is running:
   ```bash
   ollama list
   ```

2. Restart Ollama:
   ```bash
   # macOS
   killall ollama
   ollama serve &

   # Linux
   sudo systemctl restart ollama
   ```

3. Pull with timeout:
   ```bash
   timeout 1800 ollama pull llama3.2:3b
   ```

4. Use smaller model:
   ```bash
   ollama pull llama3.2:1b
   # Update config: OLLAMA_MODEL=llama3.2:1b
   ```

### Python Dependencies Fail

**Problem:** `pip install` errors

**Solutions:**
1. Upgrade pip:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. Install with verbose:
   ```bash
   pip install -r requirements.txt -v
   ```

3. Install problematic packages individually:
   ```bash
   # Common issues:
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install psycopg2-binary  # Instead of psycopg2
   ```

4. Use conda (alternative):
   ```bash
   conda create -n dheera python=3.11
   conda activate dheera
   pip install -r requirements.txt
   ```

### Database Initialization Fails

**Problem:** `init_databases.py` errors

**Solutions:**
1. Ensure containers are healthy:
   ```bash
   docker-compose ps
   # All should show "healthy" status
   ```

2. Wait longer:
   ```bash
   sleep 60  # Give databases time to start
   python3 scripts/init_databases.py
   ```

3. Check logs:
   ```bash
   docker-compose logs milvus
   docker-compose logs neo4j
   ```

4. Manual reset:
   ```bash
   docker-compose down -v
   rm -rf data/*
   docker-compose up -d
   sleep 60
   python3 scripts/init_databases.py
   ```

### Services Start But Don't Respond

**Problem:** Services running but not accessible

**Solutions:**
1. Check logs:
   ```bash
   tail -f logs/backend.log
   tail -f logs/frontend.log
   ```

2. Check ports:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8501
   ```

3. Restart services:
   ```bash
   ./stop_all.sh
   sleep 5
   ./start_all.sh
   ```

4. Check firewall:
   ```bash
   # macOS
   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

   # Linux
   sudo ufw status
   ```

---

## 🔄 Updating

### Update Dheera Code
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Update Docker Images
```bash
docker-compose pull
docker-compose up -d
```

### Update Ollama Models
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

## 🗑️ Uninstallation

### Remove Everything
```bash
# Stop services
./stop_all.sh

# Remove Docker containers and volumes
docker-compose down -v

# Remove Python environment
deactivate
rm -rf venv/

# Remove data (CAUTION: Deletes all data!)
rm -rf data/ logs/ backups/

# Uninstall Ollama
# macOS: Remove from Applications
# Linux: sudo apt remove ollama
```

### Keep Data, Remove Services
```bash
./stop_all.sh
docker-compose down  # Keep volumes
deactivate
```

---

## 📞 Support

**Issues during installation?**

1. Check logs: `logs/*.log`
2. Run verification: `./verify_installation.sh`
3. Check documentation: [docs/README.md](docs/README.md)
4. Check FAQ: [docs/troubleshooting/FAQ.md](docs/troubleshooting/FAQ.md)

**Common Issues:**
- Out of disk space → Need 20GB+ free
- Out of memory → Close other applications, need 8GB+ RAM
- Port conflicts → Check if other services using ports 8000, 8501, 19530, 7474, 6379, 5432
- Docker not running → Start Docker Desktop

---

## ✅ Installation Checklist

Use this to track your installation progress:

- [ ] Prerequisites installed (Python 3.11+, Docker, Git)
- [ ] Repository cloned
- [ ] `setup.sh` executed successfully
- [ ] Ollama models pulled (llama3.2:3b, nomic-embed-text)
- [ ] Docker containers running (7 containers)
- [ ] Python environment created (venv/)
- [ ] Dependencies installed (50+ packages)
- [ ] Databases initialized (Milvus, Neo4j, Redis, PostgreSQL)
- [ ] Configuration files created (.env, config/dheera_config.yaml)
- [ ] Verification passed (`./verify_installation.sh`)
- [ ] Services start successfully (`./start_all.sh`)
- [ ] Frontend accessible (http://localhost:8501)
- [ ] Backend accessible (http://localhost:8000)
- [ ] First chat successful

**All checked?** 🎉 **Installation complete!**

Read [QUICK_START.md](QUICK_START.md) to start using Dheera.

---

**Installation Time:** 15-30 minutes (automated) or 45-60 minutes (manual)

**Disk Usage:** ~15GB (models + databases + dependencies)

**Next:** [QUICK_START.md](QUICK_START.md) → Start using Dheera!
