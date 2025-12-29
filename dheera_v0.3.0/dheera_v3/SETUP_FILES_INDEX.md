# 📑 Dheera v0.3.0 - Complete Setup Files Index

**Package:** Full installation with databases, backend, and frontend support
**Created:** December 29, 2025
**Total Files:** 16 core files + generated files

---

## 🎯 Start Here

| File | Size | Purpose | Read When |
|------|------|---------|-----------|
| **[SETUP_PACKAGE_SUMMARY.md](SETUP_PACKAGE_SUMMARY.md)** | 12 pages | Package overview | **START HERE** |
| **[INSTALLATION.md](INSTALLATION.md)** | 15 pages | Detailed installation guide | During installation |
| **[QUICK_START.md](QUICK_START.md)** | 8 pages | Post-install quick start | After installation |

**Recommended Reading Order:**
1. SETUP_PACKAGE_SUMMARY.md (5 min read)
2. Run `./setup.sh` (15-30 min automated)
3. QUICK_START.md (3 min read)
4. Start using Dheera!

---

## 📦 Core Setup Files

### 1. Installation Scripts

#### **setup.sh** (41KB)
- **Purpose:** Main automated installation script
- **What it does:**
  - Checks system requirements (Python 3.11+, Docker)
  - Creates directory structure
  - Installs Ollama and pulls models
  - Sets up Python virtual environment
  - Installs all dependencies
  - Starts Docker containers
  - Initializes databases
  - Creates configuration files
  - Creates startup scripts
  - Runs verification tests
- **When to run:** Once, during initial installation
- **How to run:** `./setup.sh`
- **Time:** 15-30 minutes
- **Status:** ✅ Ready to use (executable)

#### **scripts/init_databases.py** (Created by setup.sh)
- **Purpose:** Initialize all databases with schemas
- **What it does:**
  - Creates Milvus collection with vector index
  - Creates Neo4j constraints and indexes
  - Clears and initializes Redis
  - Creates PostgreSQL tables (conversations, feedback, documents)
  - Converts to TimescaleDB hypertables
- **When to run:** Automatically during setup, or manually to reset databases
- **How to run:** `python3 scripts/init_databases.py`

---

### 2. Service Management Scripts

These are **created by setup.sh** during installation:

#### **start_all.sh** (Created during setup)
- **Purpose:** Start all Dheera services
- **What it starts:**
  - Docker containers (7 databases)
  - Backend API (port 8000)
  - Frontend GUI (port 8501)
- **How to run:** `./start_all.sh`
- **Logs to:** `logs/backend.log`, `logs/frontend.log`

#### **start_backend.sh** (Created during setup)
- **Purpose:** Start backend API only
- **Port:** 8000
- **How to run:** `./start_backend.sh`
- **Access:** http://localhost:8000/docs

#### **start_frontend.sh** (Created during setup)
- **Purpose:** Start frontend GUI only
- **Port:** 8501
- **How to run:** `./start_frontend.sh`
- **Access:** http://localhost:8501

#### **start_chat.sh** (Created during setup)
- **Purpose:** Start CLI chat interface
- **How to run:** `./start_chat.sh`
- **Interface:** Terminal-based

#### **stop_all.sh** (Created during setup)
- **Purpose:** Stop all running services
- **What it stops:**
  - Backend API
  - Frontend GUI
  - (Optionally) Docker containers
- **How to run:** `./stop_all.sh`

#### **verify_installation.sh** (Created during setup)
- **Purpose:** Verify all components installed correctly
- **What it checks:**
  - Python, Ollama, Docker installed
  - All 7 Docker containers running
  - Python packages importable
  - Ollama models pulled
  - Directory structure exists
- **How to run:** `./verify_installation.sh`
- **When to run:** After installation, when troubleshooting

---

### 3. Configuration Files

#### **docker-compose.yml** (Included in package)
- **Purpose:** Database services configuration
- **Defines 7 containers:**
  1. **milvus-standalone** - Vector database (ports: 19530, 9091)
  2. **milvus-etcd** - Milvus metadata store (port: 2379)
  3. **milvus-minio** - Milvus object storage (ports: 9000, 9001)
  4. **dheera-neo4j** - Knowledge graph (ports: 7474, 7687)
  5. **dheera-redis** - Cache (port: 6379)
  6. **dheera-postgresql** - Relational DB with TimescaleDB (port: 5432)
  7. **dheera-adminer** - Database web UI (port: 8080)
- **Customizable:** ✅ Ports, memory limits, volumes
- **How to use:** `docker-compose up -d`

#### **.env** (Created by setup.sh)
- **Purpose:** Environment variables
- **Contains:**
  - Database credentials (Neo4j: neo4j/dheera123, PostgreSQL: dheera/dheera123)
  - API configuration (ports, workers)
  - LLM settings (model, timeout)
  - Performance tuning (GPU, batch size, workers)
  - Security keys (change in production!)
- **Customizable:** ✅ All settings
- **Key variables:**
  ```bash
  OLLAMA_MODEL=llama3.2:3b
  SPIKING_ENABLED=true
  API_PORT=8000
  STREAMLIT_SERVER_PORT=8501
  USE_GPU=false
  ```

#### **config/dheera_config.yaml** (Updated by setup.sh)
- **Purpose:** Main Dheera configuration
- **Contains:**
  - LLM provider and model settings
  - Vector database configuration (Milvus)
  - Knowledge graph settings (Neo4j)
  - Cache configuration (Redis)
  - RAG parameters (chunk size, retrieval)
  - DQN hyperparameters
  - Spiking network settings
  - RLHF and curiosity settings
- **Customizable:** ✅ All settings
- **Key sections:**
  ```yaml
  llm:
    model: "llama3.2:3b"
    temperature: 0.7
  vector_db:
    provider: "milvus"
  spiking:
    enabled: true
  ```

---

### 4. Dependency Files

#### **requirements.txt** (Included)
- **Purpose:** Core Python dependencies
- **Contains:** Essential packages for Dheera core functionality
- **Packages:** ~25 core packages
- **When used:** By setup.sh during installation
- **How to use:** `pip install -r requirements.txt`

#### **requirements_full.txt** (Included)
- **Purpose:** All Python dependencies (all features)
- **Contains:** 50+ packages for complete functionality
- **Categories:**
  - Core ML/AI: torch, transformers, sentence-transformers
  - Databases: pymilvus, neo4j, redis, psycopg2-binary
  - API/Web: fastapi, uvicorn, streamlit
  - Document processing: PyPDF2, python-docx, pandas
  - Visualization: plotly, matplotlib, seaborn
  - Testing: pytest, black, flake8
- **When used:** By setup.sh during installation
- **How to use:** `pip install -r requirements_full.txt`

---

### 5. Documentation Files

#### **SETUP_PACKAGE_SUMMARY.md** (Included - **READ FIRST**)
- **Size:** 12 pages
- **Purpose:** Complete package overview
- **Contains:**
  - What you received
  - File manifest
  - Installation overview
  - What gets installed
  - Resource requirements
  - Verification checklist
  - Troubleshooting
  - Quick reference
- **When to read:** **Before installation**
- **Audience:** Everyone

#### **INSTALLATION.md** (Included)
- **Size:** 15 pages
- **Purpose:** Detailed installation guide
- **Contains:**
  - Prerequisites
  - Step-by-step installation (automated + manual)
  - Directory structure explanation
  - Configuration guide
  - Verification steps
  - Extensive troubleshooting (6 common issues)
  - Update and maintenance
  - Uninstallation
  - Installation checklist
- **When to read:** During installation, when troubleshooting
- **Audience:** Installers, troubleshooters

#### **QUICK_START.md** (Created by setup.sh)
- **Size:** 8 pages
- **Purpose:** Getting started after installation
- **Contains:**
  - 3-step quick start
  - Access points and URLs
  - Useful commands
  - Example interactions
  - Next steps
- **When to read:** Immediately after installation
- **Audience:** New users

#### **SETUP_README.md** (Included)
- **Size:** 10 pages
- **Purpose:** Setup package documentation
- **Contains:**
  - Package contents
  - Quick installation
  - System requirements
  - Installation output
  - Verification
  - Configuration
  - Service management
  - Common issues
  - Resource usage
- **When to read:** Reference during/after installation
- **Audience:** Installers, administrators

#### **SETUP_FILES_INDEX.md** (This file)
- **Size:** 6 pages
- **Purpose:** Complete index of all setup files
- **Contains:**
  - File descriptions
  - Purpose of each file
  - When to use each file
  - How to use each file
  - File organization
- **When to read:** To understand package structure
- **Audience:** Everyone (reference)

---

## 📚 Related Documentation

### Existing Documentation (Not Part of Setup Package)

#### **[docs/README.md](docs/README.md)**
- Main documentation hub
- Links to all guides and features
- Learning path

#### **[docs/guides/GETTING_STARTED.md](docs/guides/GETTING_STARTED.md)**
- New user guide
- Basic usage
- Examples

#### **[docs/troubleshooting/FAQ.md](docs/troubleshooting/FAQ.md)**
- Frequently asked questions
- Common problems and solutions

#### **[docs/roadmap/GOD_LEVEL_AI_PLAN.md](docs/roadmap/GOD_LEVEL_AI_PLAN.md)**
- Future development roadmap
- God-level AI architecture
- 8-week implementation plan
- 100-500x speedup targets

#### **[README.md](README.md)**
- Project overview
- Features
- Quick links

---

## 🗂️ File Organization

### Setup Package Files (Provided)
```
dheera_v3/
├── setup.sh                          ← Main installer (RUN THIS)
├── docker-compose.yml                ← Database configuration
├── .env                              ← Environment variables (created)
├── requirements.txt                  ← Core dependencies
├── requirements_full.txt             ← All dependencies
│
├── SETUP_PACKAGE_SUMMARY.md          ← Package overview (READ FIRST)
├── INSTALLATION.md                   ← Installation guide
├── QUICK_START.md                    ← Quick start (created)
├── SETUP_README.md                   ← Setup documentation
├── SETUP_FILES_INDEX.md              ← This file
│
└── config/
    └── dheera_config.yaml            ← Dheera configuration (updated)
```

### Generated During Setup
```
dheera_v3/
├── start_all.sh                      ← Start all services
├── start_backend.sh                  ← Start backend
├── start_frontend.sh                 ← Start frontend
├── start_chat.sh                     ← Start CLI
├── stop_all.sh                       ← Stop services
├── verify_installation.sh            ← Verify install
│
├── scripts/
│   └── init_databases.py             ← Database initialization
│
├── venv/                             ← Python virtual environment
├── data/                             ← Database data directories
│   ├── milvus/
│   ├── neo4j/
│   ├── redis/
│   └── postgresql/
├── logs/                             ← Application logs
├── backups/                          ← Database backups
├── checkpoints/                      ← Model checkpoints
└── uploads/                          ← Uploaded documents
```

---

## 🎯 Usage Guide by Task

### Task: Install Dheera

**Files to use:**
1. Read: [SETUP_PACKAGE_SUMMARY.md](SETUP_PACKAGE_SUMMARY.md)
2. Run: `./setup.sh`
3. Verify: `./verify_installation.sh`
4. Read: [QUICK_START.md](QUICK_START.md)

### Task: Start Services

**Files to use:**
1. Run: `./start_all.sh`
2. Check: `docker-compose ps`
3. Access: http://localhost:8501

### Task: Troubleshoot Installation

**Files to use:**
1. Run: `./verify_installation.sh`
2. Check: `tail -f logs/*.log`
3. Read: [INSTALLATION.md](INSTALLATION.md) troubleshooting section
4. Read: [docs/troubleshooting/FAQ.md](docs/troubleshooting/FAQ.md)

### Task: Customize Configuration

**Files to edit:**
1. `.env` - Environment variables
2. `config/dheera_config.yaml` - Dheera settings
3. `docker-compose.yml` - Database settings

### Task: Update Installation

**Files to use:**
```bash
git pull origin main
pip install -r requirements_full.txt --upgrade
docker-compose pull
docker-compose up -d
```

---

## 📊 File Summary Statistics

### Setup Package Files
- **Total files provided:** 16
- **Total documentation pages:** 60+
- **Total code (setup.sh):** 41KB
- **Total size:** ~50KB (excluding dependencies)

### Generated During Installation
- **Scripts generated:** 6
- **Directories created:** 8
- **Database containers:** 7
- **Python packages installed:** 50+
- **Disk usage:** ~15GB

### Documentation Coverage
- **Installation guides:** 3 files, 35 pages
- **Configuration files:** 3 files
- **Management scripts:** 6 files
- **Troubleshooting:** 2 sections, 15+ scenarios
- **Examples:** 10+ use cases

---

## ✅ Quick Reference Checklist

### Before Installation
- [ ] Read [SETUP_PACKAGE_SUMMARY.md](SETUP_PACKAGE_SUMMARY.md)
- [ ] Check system requirements (Python 3.11+, Docker, 8GB RAM, 20GB disk)
- [ ] Install prerequisites if missing

### During Installation
- [ ] Run `./setup.sh`
- [ ] Wait 15-30 minutes
- [ ] Check for errors in output

### After Installation
- [ ] Run `./verify_installation.sh`
- [ ] All checks pass
- [ ] Run `./start_all.sh`
- [ ] Access http://localhost:8501
- [ ] Read [QUICK_START.md](QUICK_START.md)
- [ ] Send test message

### Daily Use
- [ ] Start: `./start_all.sh`
- [ ] Use: http://localhost:8501
- [ ] Stop: `./stop_all.sh`

---

## 🔍 File Lookup Table

**Need to...**

| Need to... | Use this file |
|------------|---------------|
| Understand package | SETUP_PACKAGE_SUMMARY.md |
| Install Dheera | setup.sh |
| Install manually | INSTALLATION.md |
| Verify installation | verify_installation.sh |
| Start all services | start_all.sh |
| Start backend only | start_backend.sh |
| Start frontend only | start_frontend.sh |
| Start CLI chat | start_chat.sh |
| Stop services | stop_all.sh |
| Configure databases | docker-compose.yml |
| Configure environment | .env |
| Configure Dheera | config/dheera_config.yaml |
| Get started quickly | QUICK_START.md |
| Troubleshoot | INSTALLATION.md, verify_installation.sh |
| Understand files | SETUP_FILES_INDEX.md (this file) |
| Reset databases | scripts/init_databases.py |
| View logs | logs/*.log |

---

## 📞 Support & Help

**Which file to check?**

| Issue | Check This |
|-------|------------|
| Installation fails | INSTALLATION.md troubleshooting |
| Port conflicts | INSTALLATION.md → Port Configuration |
| Out of memory | SETUP_PACKAGE_SUMMARY.md → Resource Usage |
| Docker issues | INSTALLATION.md → Docker section |
| Database issues | scripts/init_databases.py logs |
| Service won't start | verify_installation.sh, logs/*.log |
| General questions | docs/troubleshooting/FAQ.md |

---

## 🎉 Summary

**Total Setup Package:** 16 core files

**Categories:**
- Installation: 2 files (setup.sh, init_databases.py)
- Service Management: 6 files (start/stop scripts)
- Configuration: 3 files (docker-compose.yml, .env, dheera_config.yaml)
- Documentation: 5 files (guides, README, index)

**Usage:**
1. **First time:** Read SETUP_PACKAGE_SUMMARY.md → Run setup.sh → Read QUICK_START.md
2. **Daily use:** ./start_all.sh → Use GUI → ./stop_all.sh
3. **Troubleshooting:** verify_installation.sh → INSTALLATION.md → FAQ

**Time to Install:** 15-30 minutes (automated)
**Time to Learn:** 10 minutes (read docs)
**Time to First Chat:** 20-40 minutes (total)

---

**Ready to get started?**

1. Read: [SETUP_PACKAGE_SUMMARY.md](SETUP_PACKAGE_SUMMARY.md) (5 min)
2. Run: `./setup.sh` (15-30 min)
3. Start: `./start_all.sh` (1 min)
4. Chat: http://localhost:8501 (instant)

**Total time to working system:** 20-40 minutes

**Enjoy Dheera!** 🧠⚡
