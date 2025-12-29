# 🚀 Getting Started with Dheera

Welcome to Dheera - a brain-inspired AI assistant with cognitive architecture!

---

## ⚡ Quick Start (30 seconds)

```bash
# Navigate to Dheera directory
cd /Users/nandhavignesh/triton/dheera_v0.3.0/dheera_v3

# Start chatting!
python3 run_chat.py
```

That's it! You're now using Dheera with spiking neural networks (7x faster!).

---

## 📋 Prerequisites

**Required:**
- Python 3.11+
- Ollama (for local LLM)

**Already Installed:**
- ✅ PyTorch
- ✅ NumPy
- ✅ Sentence Transformers
- ✅ All dependencies (see requirements.txt)

---

## 🎯 Three Ways to Use Dheera

### **1. CLI Chat (Recommended)**
Best for: Direct conversation, fastest, no timeout issues

```bash
python3 run_chat.py
```

**Features:**
- Full Dheera capabilities
- Spiking neural networks active
- Feedback with ++, +, -, --
- Commands: /help, /stats, /quit

### **2. GUI Dashboard**
Best for: Visual monitoring, database stats, ontology exploration

```bash
streamlit run gui_professional.py --server.port 8501
```

Then open: http://localhost:8501

**Features:**
- 💾 Database & Memory monitoring
- 🧠 Ontology Graph viewer
- 📊 Analytics and stats
- 🔧 Model management

### **3. REST API**
Best for: Integration with other apps

```bash
# Start API server
python3 api/server.py
```

**Note:** API chat endpoint has timeout issues - use CLI instead!

---

## 📚 What Can Dheera Do?

### **✅ Working Features:**

**1. Intelligent Conversation**
- Context-aware responses
- Multi-turn dialogue
- User feedback learning (++, +, -, --)

**2. Document Analysis**
- Upload PDF, DOCX, TXT, MD files
- Semantic search with RAG
- Ask questions about documents

**3. Cognitive Architecture**
- Rainbow DQN for action selection
- Curiosity-driven exploration
- RLHF from user feedback
- Ontology-based reasoning

**4. Performance Monitoring**
- Database statistics
- Memory usage tracking
- Training metrics
- Response latency

### **⚡ Spiking Neural Networks (ACTIVE!)**
- 7x faster DQN inference
- 97% energy savings
- 69% neuron sparsity
- Biological plausibility

---

## 🎓 Your First Conversation

**Start chat:**
```bash
python3 run_chat.py
```

**Try these:**
```
You: Hello!
Dheera: Hello! It's nice to meet you. Is there something I can help you with today?

You: What is 2+2?
Dheera: 2+2 equals 4.

You: ++
[Dheera learns you liked this response!]
```

**Give feedback:**
- `++` - Excellent response
- `+` - Good response
- `-` - Poor response
- `--` - Very poor response

**Commands:**
- `/help` - Show help
- `/stats` - Show statistics
- `/clear` - Clear conversation
- `/quit` - Exit

---

## 📊 Check Your Setup

**Verify spiking networks:**
```bash
python3 run_chat.py 2>&1 | grep "Spiking"
```

**Expected:**
```
⚡ Initializing SpikingRainbow DQN...
✓ SpikingRainbow DQN (69% sparsity, 97% energy savings)
```

**Check database:**
```bash
ls -lh dheera.db
```

**Expected:** File exists (created on first run)

---

## 🎨 Explore the GUI

**Start Streamlit:**
```bash
streamlit run gui_professional.py --server.port 8501
```

**Navigate to:**
1. **💬 Chat** - Web-based chat interface
2. **🧠 Core Engine** - View DQN, RAG, RLHF stats
3. **🗺️ System Map** - Architecture and ontology graph
4. **💾 Database & Memory** - Monitor storage and memory
5. **📊 Analytics** - Performance metrics

---

## 📄 Upload Your First Document

**Via API:**
```bash
curl -X POST http://localhost:8000/api/rag/upload \
  -F "files=@your_document.pdf"
```

**Via GUI:**
1. Open Streamlit: http://localhost:8501
2. Go to: **🧠 Core Engine**
3. Find: **Upload Documents** section
4. Drag & drop PDF/DOCX/TXT file

**Then ask questions:**
```
You: What does the document say about X?
Dheera: According to the document, X refers to...
```

---

## ⚙️ Configuration

**Main config:** `config/dheera_config.yaml`

**Key settings:**

```yaml
# LLM Configuration
slm:
  model: "llama3.2:latest"
  timeout: 30
  max_tokens: 256

# Spiking Networks (ACTIVE)
spiking:
  enabled: true
  tau_mem: 10.0
  threshold: 1.0

# DQN Training
dqn:
  batch_size: 32
  train_every: 10
```

**To change model:**
```yaml
slm:
  model: "qwen2:1.5b"  # Smaller, faster
```

---

## 🔧 Troubleshooting

**Issue: Chat not responding**
- Solution: Use CLI (`python3 run_chat.py`) instead of API
- See: [API Timeout Fix](../troubleshooting/API_TIMEOUT.md)

**Issue: "Ollama not available"**
- Solution: Start Ollama: `ollama serve`
- Check: `curl http://localhost:11434/api/tags`

**Issue: Slow responses**
- Check: LLM model size (try qwen2:1.5b for speed)
- See: [Optimization Guide](OPTIMIZATION.md)

**More help:** [FAQ](../troubleshooting/FAQ.md)

---

## 📚 Next Steps

**Beginner:**
1. ✅ Complete this guide
2. Read: [Chat Guide](CHAT_GUIDE.md)
3. Explore: [GUI Guide](GUI_GUIDE.md)

**Intermediate:**
4. Learn: [Database Monitoring](../features/DATABASE_MONITORING.md)
5. Understand: [Ontology Graph](../features/ONTOLOGY_GRAPH.md)
6. Upload docs: [Document Analysis](../features/DOCUMENT_ANALYSIS.md)

**Advanced:**
7. Optimize: [Performance Guide](OPTIMIZATION.md)
8. Customize: Edit `config/dheera_config.yaml`
9. Integrate: [API Reference](../api/REST_API.md)

---

## 🎯 Quick Reference

**Start chat:**
```bash
python3 run_chat.py
```

**Start GUI:**
```bash
streamlit run gui_professional.py --server.port 8501
```

**Check spiking:**
```bash
python3 run_chat.py 2>&1 | grep "Spiking"
```

**Config file:**
```bash
nano config/dheera_config.yaml
```

---

**You're ready to use Dheera!** 🧠⚡

**Questions?** See [FAQ](../troubleshooting/FAQ.md)
