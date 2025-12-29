# 🛠️ Dheera v0.3.0 - Complete Setup Package

This package contains everything needed to install and run Dheera with full database support.

---

## 📦 Package Contents

### Installation Scripts
- **[setup.sh](setup.sh)** - Main automated installation script (15-30 min)
- **[verify_installation.sh](verify_installation.sh)** - Verify all components installed correctly
- **[scripts/init_databases.py](scripts/init_databases.py)** - Initialize all databases

### Service Management
- **[start_all.sh](start_all.sh)** - Start all services (databases + backend + frontend)
- **[start_backend.sh](start_backend.sh)** - Start backend API only (port 8000)
- **[start_frontend.sh](start_frontend.sh)** - Start frontend GUI only (port 8501)
- **[start_chat.sh](start_chat.sh)** - Start CLI chat interface
- **[stop_all.sh](stop_all.sh)** - Stop all running services

### Configuration Files
- **[docker-compose.yml](docker-compose.yml)** - Database services configuration
  - Milvus (vector database)
  - Neo4j (knowledge graph)
  - Redis (cache)
  - PostgreSQL + TimescaleDB (relational + time-series)
  - Adminer (database web UI)

- **[.env](.env)** - Environment variables
  - Database credentials
  - API settings
  - Performance tuning
  - Model configuration

- **[config/dheera_config.yaml](config/dheera_config.yaml)** - Main Dheera configuration
  - LLM settings
  - Vector DB settings
  - Spiking networks
  - RAG configuration
  - DQN parameters

### Dependencies
- **[requirements.txt](requirements.txt)** - Core Python dependencies
- **[requirements_full.txt](requirements_full.txt)** - All features (50+ packages)

### Documentation
- **[INSTALLATION.md](INSTALLATION.md)** - Complete installation guide with troubleshooting
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide after installation
- **[docs/README.md](docs/README.md)** - Main documentation hub

---

## 🚀 Quick Installation

### One-Command Install
```bash
chmod +x setup.sh && ./setup.sh
```

**What it does:**
1. ✅ Checks system requirements (Python 3.11+, Docker, Git)
2. ✅ Creates directory structure (data/, logs/, backups/)
3. ✅ Installs Ollama and pulls models (llama3.2:3b, nomic-embed-text)
4. ✅ Sets up Python virtual environment
5. ✅ Installs all Python dependencies (50+ packages)
6. ✅ Starts Docker containers (7 database services)
7. ✅ Initializes all databases with schemas
8. ✅ Creates configuration files (.env, configs)
9. ✅ Creates startup scripts
10. ✅ Runs verification tests

**Time:** 15-30 minutes (depending on internet speed)

### After Installation
```bash
./start_all.sh              # Start all services
open http://localhost:8501  # Open web GUI
```

---

## 📋 System Requirements

### Minimum Requirements
- **OS:** macOS 10.15+ or Ubuntu 20.04+
- **RAM:** 8GB (16GB recommended)
- **Disk:** 20GB free space (50GB recommended)
- **CPU:** 4 cores (8+ recommended)
- **Python:** 3.11+
- **Docker:** Latest version
- **Internet:** Required for installation

### Before Installation
Install these first:

1. **Python 3.11+**
   ```bash
   # macOS
   brew install python@3.11

   # Ubuntu
   sudo apt install python3.11 python3.11-venv
   ```

2. **Docker Desktop** (macOS) or **Docker + Docker Compose** (Linux)
   ```bash
   # macOS: Download from https://www.docker.com/products/docker-desktop

   # Ubuntu
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

3. **Git** (usually pre-installed)
   ```bash
   git --version
   ```

---

## 🗂️ Installation Output

### Directory Structure Created
```
dheera_v3/
├── setup.sh                    ← Run this first
├── docker-compose.yml          ← Database services
├── .env                        ← Environment config
│
├── venv/                       ← Python environment (created)
│
├── data/                       ← Database data (created)
│   ├── milvus/                 ← Vector database
│   ├── neo4j/                  ← Knowledge graph
│   ├── redis/                  ← Cache
│   └── postgresql/             ← Relational database
│
├── logs/                       ← Application logs (created)
│   ├── backend.log
│   ├── frontend.log
│   └── dheera.log
│
├── backups/                    ← Database backups (created)
├── checkpoints/                ← Model checkpoints (created)
├── uploads/                    ← Uploaded files (created)
│
├── start_all.sh                ← Start everything (created)
├── start_backend.sh            ← Backend only (created)
├── start_frontend.sh           ← Frontend only (created)
├── start_chat.sh               ← CLI chat (created)
├── stop_all.sh                 ← Stop services (created)
│
└── verify_installation.sh      ← Verify install (created)
```

### Docker Containers Started

| Container | Purpose | Port(s) | Status |
|-----------|---------|---------|--------|
| milvus-standalone | Vector database | 19530, 9091 | ✅ Running |
| milvus-etcd | Milvus metadata | 2379 | ✅ Running |
| milvus-minio | Milvus storage | 9000, 9001 | ✅ Running |
| dheera-neo4j | Knowledge graph | 7474, 7687 | ✅ Running |
| dheera-redis | Cache | 6379 | ✅ Running |
| dheera-postgresql | Relational DB | 5432 | ✅ Running |
| dheera-adminer | DB web UI | 8080 | ✅ Running |

### Python Packages Installed (50+)

**Categories:**
- Core: torch, numpy, scipy, pyyaml (4)
- Databases: pymilvus, chromadb, neo4j, redis, psycopg2 (8)
- LLM: sentence-transformers, transformers, openai, anthropic (7)
- API: fastapi, uvicorn, streamlit, websockets (8)
- Documents: PyPDF2, python-docx, pandas, Pillow (7)
- Visualization: plotly, matplotlib, seaborn, networkx (6)
- Testing: pytest, black, flake8, mypy (6)
- Utils: tqdm, rich, loguru, python-dotenv (10+)

---

## ✅ Verification

### Automatic Verification
```bash
./verify_installation.sh
```

**Checks:**
- ✅ Python 3.11+ installed
- ✅ Ollama installed and running
- ✅ Docker installed
- ✅ All 7 Docker containers running and healthy
- ✅ All Python packages importable
- ✅ Ollama models pulled (llama3.2:3b, nomic-embed-text)
- ✅ Directory structure created
- ✅ Configuration files exist

### Manual Verification
```bash
# Check Python
python3 --version  # Should be 3.11+

# Check Docker containers
docker-compose ps  # All should show "Up" and "healthy"

# Check Ollama
ollama list  # Should show llama3.2:3b and nomic-embed-text

# Test databases
python3 -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('✅ Milvus OK')"
python3 -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'dheera123')); driver.verify_connectivity(); print('✅ Neo4j OK')"
python3 -c "import redis; r = redis.Redis(host='localhost'); r.ping(); print('✅ Redis OK')"
```

---

## 🎯 Access Points

After running `./start_all.sh`:

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Frontend GUI** | http://localhost:8501 | - | Main web interface |
| **Backend API** | http://localhost:8000 | - | REST API |
| **API Docs** | http://localhost:8000/docs | - | Interactive API docs |
| **Neo4j Browser** | http://localhost:7474 | neo4j / dheera123 | Knowledge graph explorer |
| **Database Admin** | http://localhost:8080 | - | PostgreSQL admin |
| **Milvus** | localhost:19530 | - | Vector DB (API only) |
| **Redis** | localhost:6379 | - | Cache (API only) |

---

## 🔧 Configuration

### Environment Variables (.env)

Key settings you can customize:

```bash
# LLM Model
OLLAMA_MODEL=llama3.2:3b          # Change to llama3.2:1b for lower memory

# Performance
SPIKING_ENABLED=true              # Enable spiking networks (7x faster)
USE_GPU=false                     # Set to true if you have NVIDIA GPU
MAX_WORKERS=4                     # Increase if you have more CPU cores

# API
API_PORT=8000                     # Change if port conflict
STREAMLIT_SERVER_PORT=8501        # Change if port conflict

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=your-secret-key-change-this-in-production
```

### Dheera Config (config/dheera_config.yaml)

Key settings:

```yaml
llm:
  model: "llama3.2:3b"           # LLM model to use
  temperature: 0.7                # Lower = more focused, higher = more creative
  max_tokens: 512                 # Response length

spiking:
  enabled: true                   # Enable spiking networks (7x faster DQN)

rag:
  default_n_results: 3            # Number of documents to retrieve
  chunk_size: 500                 # Document chunk size

dqn:
  batch_size: 32                  # Training batch size
  train_every: 10                 # Training frequency
```

---

## 🚦 Service Management

### Start Services
```bash
# Start everything (recommended)
./start_all.sh

# Or start individually:
docker-compose up -d        # Start databases only
./start_backend.sh          # Start backend API
./start_frontend.sh         # Start web GUI
./start_chat.sh             # Start CLI chat
```

### Stop Services
```bash
# Stop all
./stop_all.sh

# Or individually:
kill $(cat logs/backend.pid)    # Stop backend
kill $(cat logs/frontend.pid)   # Stop frontend
docker-compose down             # Stop databases
```

### View Logs
```bash
# Application logs
tail -f logs/backend.log
tail -f logs/frontend.log
tail -f logs/dheera.log

# Database logs
docker-compose logs -f
docker-compose logs -f milvus
docker-compose logs -f neo4j
```

### Restart Services
```bash
./stop_all.sh && sleep 5 && ./start_all.sh
```

---

## 🐛 Common Issues & Solutions

### 1. Setup Script Fails

**Problem:** `setup.sh` exits with error

**Solution:**
```bash
# Check requirements
python3 --version  # Must be 3.11+
docker --version   # Must be installed

# Run in debug mode
bash -x setup.sh 2>&1 | tee setup.log

# Check disk space (need 20GB+)
df -h .
```

### 2. Docker Containers Won't Start

**Problem:** `docker-compose up -d` fails

**Solution:**
```bash
# Check Docker is running
docker ps

# Increase Docker resources
# Docker Desktop → Settings → Resources
# Set: 8GB RAM, 4 CPUs minimum

# Clean and restart
docker-compose down -v
docker system prune -a
docker-compose up -d
```

### 3. Port Already in Use

**Problem:** "Port 8000 already in use"

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill it
kill -9 <PID>

# Or change port in .env
nano .env
# Change: API_PORT=8001
```

### 4. Ollama Models Won't Pull

**Problem:** `ollama pull` hangs or fails

**Solution:**
```bash
# Restart Ollama
killall ollama
ollama serve &

# Pull with smaller model first
ollama pull llama3.2:1b

# Update .env
nano .env
# Change: OLLAMA_MODEL=llama3.2:1b
```

### 5. Out of Memory

**Problem:** Services crash or become unresponsive

**Solution:**
```bash
# Use smaller model
OLLAMA_MODEL=llama3.2:1b

# Reduce batch size in config/dheera_config.yaml
dqn:
  batch_size: 16  # Instead of 32

# Reduce max_tokens
llm:
  max_tokens: 256  # Instead of 512

# Close other applications
```

### 6. Database Connection Errors

**Problem:** "Connection refused" or "Cannot connect to database"

**Solution:**
```bash
# Wait for containers to be healthy
sleep 60

# Check container health
docker-compose ps

# Restart databases
docker-compose restart

# Reinitialize
python3 scripts/init_databases.py
```

---

## 📊 Resource Usage

### Disk Space
- **Installation:** ~15GB total
  - Ollama models: 2-4GB
  - Python packages: 1-2GB
  - Docker images: 5-8GB
  - Data directories: 1-2GB (grows with use)

### Memory Usage
- **Idle:** 2-3GB
- **Running (no GPU):** 4-6GB
  - Databases: 2-3GB
  - Backend: 1-2GB
  - Frontend: 500MB-1GB
  - LLM inference: 2-3GB (per query)

- **Running (with GPU):** 8-12GB
  - GPU memory: 4-6GB additional

### CPU Usage
- **Idle:** 5-10%
- **During inference:** 50-200% (multi-core)
- **During training:** 200-400% (multi-core)

---

## 🔄 Updating

### Update Code
```bash
git pull origin main
pip install -r requirements_full.txt --upgrade
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

# Remove Docker containers and data
docker-compose down -v
rm -rf data/

# Remove Python environment
deactivate
rm -rf venv/

# Remove logs and backups
rm -rf logs/ backups/ checkpoints/

# Uninstall Ollama (optional)
# macOS: Remove from Applications
# Linux: sudo apt remove ollama
```

### Keep Data, Remove Services Only
```bash
./stop_all.sh
docker-compose down  # Keeps volumes
deactivate
```

---

## 📞 Getting Help

### Documentation
1. [INSTALLATION.md](INSTALLATION.md) - Detailed installation guide
2. [QUICK_START.md](QUICK_START.md) - Getting started after installation
3. [docs/README.md](docs/README.md) - Complete documentation
4. [docs/troubleshooting/FAQ.md](docs/troubleshooting/FAQ.md) - FAQ

### Logs
Check these for errors:
```bash
tail -f logs/backend.log      # Backend errors
tail -f logs/frontend.log     # Frontend errors
tail -f logs/dheera.log       # Application errors
docker-compose logs -f        # Database errors
```

### Support Channels
- Check logs first
- Run `./verify_installation.sh`
- See [INSTALLATION.md](INSTALLATION.md) troubleshooting section
- See [docs/troubleshooting/FAQ.md](docs/troubleshooting/FAQ.md)

---

## ✅ Installation Checklist

Track your setup progress:

**Prerequisites:**
- [ ] Python 3.11+ installed
- [ ] Docker Desktop installed (macOS) or Docker + Docker Compose (Linux)
- [ ] Git installed
- [ ] 20GB+ disk space available
- [ ] 8GB+ RAM available

**Installation:**
- [ ] Run `./setup.sh` successfully
- [ ] Ollama installed and models pulled
- [ ] 7 Docker containers running
- [ ] Python virtual environment created
- [ ] 50+ Python packages installed
- [ ] Databases initialized
- [ ] Configuration files created
- [ ] Startup scripts created

**Verification:**
- [ ] Run `./verify_installation.sh` - all checks pass
- [ ] Run `./start_all.sh` - services start
- [ ] Open http://localhost:8501 - GUI loads
- [ ] Open http://localhost:8000/docs - API docs load
- [ ] Send test message - get response

**All checked?** 🎉 Installation complete!

---

## 🎯 Next Steps

1. **Read Quick Start:** [QUICK_START.md](QUICK_START.md)
2. **Start Services:** `./start_all.sh`
3. **Open GUI:** http://localhost:8501
4. **Start Chatting!**

---

## 📄 File Manifest

Setup package includes:

**Scripts (8):**
- setup.sh (main installer)
- verify_installation.sh
- start_all.sh
- start_backend.sh
- start_frontend.sh
- start_chat.sh
- stop_all.sh
- scripts/init_databases.py

**Configuration (4):**
- docker-compose.yml
- .env
- config/dheera_config.yaml
- requirements_full.txt

**Documentation (4):**
- INSTALLATION.md
- QUICK_START.md
- SETUP_README.md (this file)
- docs/README.md

**Total:** 16 files for complete setup

---

**Setup Time:** 15-30 minutes

**Disk Usage:** ~15GB

**Memory Required:** 8GB minimum, 16GB recommended

**Ready to install?** Run: `chmod +x setup.sh && ./setup.sh`

**Enjoy Dheera!** 🧠⚡
