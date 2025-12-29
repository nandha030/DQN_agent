# 💾 Database & Memory Monitoring - Complete Guide

## 🎉 Feature Overview

Comprehensive monitoring and management of Dheera's storage systems:
- **SQLite Database**: Experience replay, episodes, turns, RLHF preferences
- **ChromaDB Vector Store**: RAG documents and embeddings
- **System Memory**: Process and system-wide memory usage
- **Cleanup Tools**: Optimize and maintain database health

---

## 📊 What's Being Tracked

### **1. SQLite Database (dheera.db)**

#### **Tables:**

| Table | Purpose | Typical Size |
|-------|---------|--------------|
| **experiences** | Rainbow DQN experience replay buffer | 10K-100K records |
| **episodes** | Conversation episodes metadata | 10-1000 records |
| **turns** | Individual conversation turns | 100-10K records |
| **preferences** | RLHF preference pairs (chosen vs rejected) | 0-1K records |
| **curiosity_states** | State visit tracking for novelty detection | 100-10K states |
| **embeddings** | Backup of embeddings | 100-1K vectors |
| **search_cache** | Cached web search results (TTL: 5min) | 10-100 entries |
| **analytics** | Performance metrics | 100-10K metrics |
| **model_checkpoints** | Model save metadata | 10-100 checkpoints |
| **reward_model_data** | RLHF reward model training data | 100-1K samples |
| **cognitive_cache** | Cached cognitive analysis results | 100-1K entries |
| **user_profiles** | User-specific settings | 1-10 profiles |

#### **Current Statistics** (based on your data):
- 📊 **Experiences**: 120 (DQN training samples)
- 💬 **Episodes**: 26 (conversation sessions)
- 💬 **Turns**: 146 (individual messages)
- 👍 **Preferences**: 0 (RLHF pairs)
- 🔥 **Curiosity States**: 0 (novel states)
- 🧬 **Embeddings**: 0 (backup embeddings)
- ⚡ **Search Cache**: 8 (cached searches)

---

### **2. ChromaDB Vector Store (chroma_db/)**

#### **Purpose:**
- Stores RAG document embeddings
- Semantic search for context retrieval
- 384-dimensional vectors (mxbai-embed-large model)

#### **Collections:**
- `dheera_memory`: Main knowledge base
- `dheera_rag`: RAG documents (if exists)
- Custom collections per use case

#### **Current Status:**
- **Directory Size**: 0 MB (not initialized yet)
- **Documents**: Upload files in Core Engine to populate

---

### **3. Memory Usage**

#### **Process Memory (Streamlit GUI):**
- **RSS (Resident Set Size)**: Actual RAM used
- **VMS (Virtual Memory Size)**: Virtual memory allocated
- **CPU Usage**: Percentage of CPU time
- **Thread Count**: Active threads

#### **Estimated Component Memory:**
| Component | Estimated Size | Notes |
|-----------|----------------|-------|
| Rainbow DQN Networks | 50-100 MB | Neural network weights |
| Experience Replay Buffer | 10-50 MB | In-memory experiences |
| RAG Embeddings Cache | 20-50 MB | Embedding model + cache |
| ChromaDB | Disk-based | No RAM overhead when idle |
| Curiosity Module (ICM) | 30-50 MB | Forward/inverse models |
| Session State | 5-10 MB | Conversation history |

**Total Estimated**: 115-260 MB for typical operation

---

## 🎯 How to Use the Database & Memory Page

### **Access:**
Navigate to: **💾 Database & Memory** in the sidebar

---

## 📋 Tab 1: SQLite Database

### **Features:**

#### **1. Database Overview**
- **Database Size**: Total file size in MB
- **Database File**: Path to dheera.db
- **Last Modified**: Timestamp of last write

#### **2. Table Record Counts**
Visual grid showing record counts for all tables with icons:
- 🧠 Experience tables
- 💬 Conversation tables (episodes, turns)
- 👍 RLHF tables (preferences, reward data)
- 🔥 Curiosity tables
- 🧬 Embedding tables
- ⚡ Cache tables

#### **3. Detailed Table Analysis**
Select any table to view:
- **Schema**: Column names, types, constraints, primary keys
- **Sample Records**: Last 10 records (truncated for readability)
- **Data Preview**: Pandas DataFrame view

### **Example Usage:**

**Check Experience Buffer Size:**
1. Go to Database tab
2. Look at "🧠 experiences" metric
3. If >100K: Consider cleanup

**View Recent Conversations:**
1. Select "turns" table
2. View last 10 turns
3. See user messages, responses, actions, rewards

**Analyze RLHF Data:**
1. Select "preferences" table
2. Check approved vs rejected responses
3. Monitor preference strength

---

## 🧬 Tab 2: Vector Store (ChromaDB)

### **Features:**

#### **1. Storage Overview**
- **Vector Store Size**: Total ChromaDB directory size
- **Total Files**: Number of files in chroma_db/
- **Storage Path**: Directory location

#### **2. RAG Collection Stats** (from backend)
- **Total Documents**: Uploaded documents
- **Total Chunks**: Document chunks for retrieval
- **Total Queries**: RAG queries performed
- **Avg Relevance**: Average similarity score

#### **3. Collection Browser**
Browse all ChromaDB collections:
- Collection name
- Document count
- Sample document IDs

### **Example Usage:**

**Check RAG Storage:**
1. Go to Vector Store tab
2. Look at "Vector Store Size"
3. View document count

**Monitor Retrieval Performance:**
1. Check "Avg Relevance" metric
2. If <0.5: Documents may be low quality
3. If 0.7+: Good retrieval quality

**Browse Uploaded Documents:**
1. Expand collection (e.g., "dheera_memory")
2. View sample document IDs
3. Verify uploads successful

---

## 💻 Tab 3: Memory Usage

### **Features:**

#### **1. Process Memory**
Current Streamlit/Dheera process:
- **Process Memory (RSS)**: Actual RAM usage
- **Virtual Memory**: Total allocated
- **CPU Usage**: Real-time CPU%
- **Thread Count**: Active threads

#### **2. System-Wide Memory**
Overall system resources:
- **Total RAM**: System total
- **Available RAM**: Free RAM
- **Used RAM**: Currently in use
- **RAM Usage**: Visual progress bar

#### **3. Component Memory Estimates**
Breakdown of memory usage by component:
- DQN networks
- Experience buffer
- RAG cache
- ChromaDB
- Curiosity module
- Session state

### **Example Usage:**

**Check if System is Overloaded:**
1. Go to Memory Usage tab
2. Check "RAM Usage" progress bar
3. If >90%: Close other applications
4. If >95%: Consider reducing buffer sizes

**Identify Memory Leaks:**
1. Monitor "Process Memory (RSS)"
2. Send 10-20 messages
3. Check if memory keeps growing
4. If yes: Potential memory leak

**Optimize Performance:**
1. Check "CPU Usage"
2. If >80% constantly: Reduce DQN training frequency
3. If high thread count: Check for stuck processes

---

## 🧹 Tab 4: Cleanup & Optimize

### **Features:**

#### **1. Cleanup Operations**

**Experience Buffer Cleanup:**
- Set max experience count (default: 100,000)
- Deletes oldest experiences
- Keeps most recent by priority + timestamp

**Search Cache Cleanup:**
- Removes expired search results
- TTL: 5 minutes (configurable)
- Frees up database space

#### **2. Database Optimization**

**VACUUM Command:**
- Rebuilds database file
- Reclaims unused space
- Improves query performance
- Shows before/after size

#### **3. Dangerous Operations** (⚠️ Use with Caution!)

**Clear All Experiences:**
- Deletes entire experience replay buffer
- DQN training starts fresh
- Cannot be undone!

**Clear All Episodes & Turns:**
- Deletes all conversation history
- Fresh start for episodes
- Cannot be undone!

#### **4. Backup Database**
- Creates timestamped backup: `dheera_backup_YYYYMMDD_HHMMSS.db`
- Full database copy
- Safe to restore from

### **Example Usage:**

**Regular Maintenance (Weekly):**
1. Click "⚡ VACUUM Database"
2. Check space savings
3. Click "🧹 Clear Expired Search Cache"

**Limit Experience Buffer:**
1. Set max to 50,000
2. Click "🧹 Cleanup Old Experiences"
3. Keeps most important 50K experiences

**Before Major Changes:**
1. Click "📦 Create Backup"
2. Note backup filename
3. Make changes safely

**Fresh Start:**
1. Click "📦 Create Backup" (safety!)
2. Expand "Nuclear Options"
3. Check "I understand..."
4. Click appropriate clear button

---

## 📈 Monitoring Best Practices

### **Daily Monitoring:**
- Check process memory: Should be <500 MB
- Verify backend status: Should be online
- Check RAM usage: Should be <80%

### **Weekly Maintenance:**
- VACUUM database
- Clear expired cache
- Review experience buffer size
- Check ChromaDB size

### **Monthly Cleanup:**
- Limit experiences to 50K-100K
- Archive old episodes (if needed)
- Create backup
- Review table sizes

### **Performance Thresholds:**

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Database Size | <100 MB | 100-500 MB | >500 MB |
| Experience Count | <50K | 50K-100K | >100K |
| RAM Usage | <60% | 60-85% | >85% |
| Process Memory | <300 MB | 300-700 MB | >700 MB |

---

## 🔧 Troubleshooting

### **Problem: Database is too large (>500 MB)**

**Solution:**
1. Go to Cleanup tab
2. Limit experiences to 50,000
3. Clear expired cache
4. VACUUM database
5. Expected reduction: 30-70%

---

### **Problem: "Database is locked" error**

**Solution:**
1. Close other Dheera instances
2. Check for zombie processes: `ps aux | grep python3`
3. Restart backend and GUI
4. If persists: Backup + delete database + restart

---

### **Problem: High memory usage (>1 GB)**

**Solution:**
1. Check Memory Usage tab
2. Identify component using most RAM
3. If Experience Buffer: Reduce buffer size
4. If DQN: Reduce batch size in config
5. If RAG: Clear embedding cache
6. Restart services

---

### **Problem: ChromaDB not found**

**Solution:**
1. Install: `pip install chromadb`
2. Restart services
3. Upload a document in Core Engine
4. ChromaDB will initialize automatically

---

### **Problem: Cannot access database**

**Solution:**
1. Check file permissions: `ls -l dheera.db`
2. Should be readable/writable
3. If not: `chmod 644 dheera.db`
4. Check ownership: Should be your user

---

## 📊 Database Schema Reference

### **Experiences Table** (Rainbow DQN)
```sql
CREATE TABLE experiences (
    id INTEGER PRIMARY KEY,
    state BLOB,                    -- 64-dim state vector
    action INTEGER,                -- Action ID (0-7)
    reward REAL,                   -- Immediate reward
    next_state BLOB,               -- Next state vector
    done INTEGER,                  -- Episode terminal flag
    priority REAL,                 -- Prioritized replay priority
    intrinsic_reward REAL,         -- Curiosity bonus
    novelty_score REAL,            -- Novelty metric
    episode_id TEXT,               -- Associated episode
    n_step_reward REAL,            -- N-step return
    n_step_next_state BLOB,        -- N-step next state
    n_step INTEGER,                -- N-step value
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Episodes Table**
```sql
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,           -- UUID
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    user_id TEXT,
    summary TEXT,
    turn_count INTEGER,
    total_reward REAL,
    total_intrinsic_reward REAL,
    search_count INTEGER,
    tool_count INTEGER,
    action_distribution TEXT,      -- JSON
    metadata TEXT                  -- JSON
);
```

### **Turns Table**
```sql
CREATE TABLE turns (
    id INTEGER PRIMARY KEY,
    episode_id TEXT,
    turn_number INTEGER,
    user_message TEXT,
    assistant_response TEXT,
    action_id INTEGER,
    action_name TEXT,
    state_vector BLOB,
    intent TEXT,
    intent_confidence REAL,
    entities TEXT,                 -- JSON
    dialogue_state TEXT,           -- JSON
    immediate_reward REAL,
    intrinsic_reward REAL,
    rag_context TEXT,
    rag_sources TEXT,              -- JSON
    search_performed INTEGER,
    search_query TEXT,
    search_results TEXT,           -- JSON
    tool_used TEXT,
    tool_input TEXT,
    tool_output TEXT,
    latency_ms REAL,
    tokens_used INTEGER,
    human_feedback REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🎯 Summary

**What You Can Now Do:**
- ✅ Monitor SQLite database size and record counts
- ✅ Browse all 12 tables with detailed schema and data
- ✅ Track ChromaDB vector store and RAG documents
- ✅ Monitor process and system memory usage in real-time
- ✅ Clean up old experiences and expired cache
- ✅ Optimize database with VACUUM command
- ✅ Create backups before risky operations
- ✅ Clear specific data with safety confirmations
- ✅ View component-wise memory estimates
- ✅ Analyze individual tables with sample data

**Key Benefits:**
- 🔍 Full visibility into all storage systems
- 🧹 Prevent database bloat
- ⚡ Optimize performance
- 💾 Safe backup and restore
- 📊 Monitor memory usage patterns
- 🎯 Identify performance bottlenecks

**Status:** ✅ **FULLY IMPLEMENTED - READY TO USE**

Restart Streamlit to access the new page!

---

🎉 **Enjoy complete control over Dheera's storage and memory!**
