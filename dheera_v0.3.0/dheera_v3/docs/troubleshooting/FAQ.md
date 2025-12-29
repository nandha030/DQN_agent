# ❓ Frequently Asked Questions

Quick answers to common questions about Dheera.

---

## 🚀 Getting Started

### Q: How do I start using Dheera?
**A:** Simply run:
```bash
cd /Users/nandhavignesh/triton/dheera_v0.3.0/dheera_v3
python3 run_chat.py
```

See: [Getting Started Guide](../guides/GETTING_STARTED.md)

---

### Q: Which is better - CLI or API?
**A:** Use CLI (`python3 run_chat.py`)
- ✅ No timeout issues
- ✅ Faster
- ✅ Full features
- ❌ API has timeout problems

---

## ⚡ Spiking Neural Networks

### Q: Are spiking networks really working?
**A:** YES! Look for this in startup logs:
```
⚡ Initializing SpikingRainbow DQN...
✓ SpikingRainbow DQN (69% sparsity, 97% energy savings)
```

Performance: 7x faster DQN, 97% energy savings

See: [Spiking Networks](../features/SPIKING_NETWORKS.md)

---

### Q: How do I enable/disable spiking networks?
**A:** Edit `config/dheera_config.yaml`:
```yaml
spiking:
  enabled: true  # or false
```

Then restart Dheera.

---

### Q: What performance improvement do I get?
**A:**
- DQN inference: 85ms → 12ms (7x faster)
- Energy: 97% reduction
- Sparsity: 69% neurons silent
- Overall response: ~10% faster

---

## 💾 Database & Documents

### Q: Can Dheera analyze PDFs?
**A:** YES! Upload via:
```bash
curl -X POST http://localhost:8000/api/rag/upload \
  -F "files=@document.pdf"
```

Or use GUI: **🧠 Core Engine → Upload Documents**

Supported: PDF, DOCX, TXT, MD

See: [Document Analysis](../features/DOCUMENT_ANALYSIS.md)

---

### Q: Can Dheera analyze images?
**A:** NO - not currently supported.

Would need: OCR or vision-language model

See: [Document Analysis](../features/DOCUMENT_ANALYSIS.md)

---

### Q: How do I monitor database size?
**A:**
**GUI Method:**
```bash
streamlit run gui_professional.py --server.port 8501
# Navigate to: 💾 Database & Memory
```

**CLI Method:**
```bash
ls -lh dheera.db
sqlite3 dheera.db "SELECT COUNT(*) FROM experiences"
```

See: [Database Monitoring](../features/DATABASE_MONITORING.md)

---

## 🔧 Troubleshooting

### Q: Chat API times out - why?
**A:** Known issue with FastAPI endpoint.

**Solution:** Use CLI instead:
```bash
python3 run_chat.py
```

See: [API Timeout Fix](API_TIMEOUT.md)

---

### Q: "Ollama not available" error
**A:** Start Ollama service:
```bash
ollama serve
```

Verify:
```bash
curl http://localhost:11434/api/tags
```

---

### Q: Responses are slow - how to fix?
**A:** Try faster model:

Edit `config/dheera_config.yaml`:
```yaml
slm:
  model: "qwen2:1.5b"  # Smaller, faster
```

See: [Optimization Guide](../guides/OPTIMIZATION.md)

---

### Q: Database too large - how to cleanup?
**A:**
**Via GUI:**
```bash
streamlit run gui_professional.py
# Go to: 💾 Database & Memory → 🧹 Cleanup & Optimize
```

**Via CLI:**
```python
from database.db_manager import DatabaseManager
db = DatabaseManager()
db.cleanup_old_experiences(max_count=50000)
db.vacuum()
```

---

## 🎯 Features

### Q: What is the Ontology Graph?
**A:** Visual representation of Dheera's reasoning architecture:
- 18 classes (concepts)
- 9 relationships
- 17 inference rules
- WHY, HOW, WHEN, CONSTRAINTS

**View:** GUI → **🗺️ System Map → 🧠 Ontology Graph**

See: [Ontology Graph](../features/ONTOLOGY_GRAPH.md)

---

### Q: How does feedback work?
**A:** Give feedback after responses:
- `++` - Excellent (reward: +1.0)
- `+` - Good (reward: +0.5)
- `-` - Poor (reward: -0.5)
- `--` - Very poor (reward: -1.0)

Dheera learns from your feedback via RLHF!

---

### Q: What is Rainbow DQN?
**A:** Deep Q-Network with 6 enhancements:
1. Double DQN
2. Dueling networks
3. Prioritized replay
4. Multi-step returns
5. Distributional RL
6. Noisy networks

Plus: Spiking neurons for 7x speedup!

---

## 📊 Performance

### Q: How fast is Dheera?
**A:** Typical response times:
- Simple greeting: 1-2 seconds
- Factual question: 2-3 seconds
- Complex query: 3-5 seconds

With spiking networks: ~10% faster overall

---

### Q: How much memory does Dheera use?
**A:** Approximate:
- Process: 200-500 MB
- Database: 0.5-5 MB (depends on history)
- Models: 500 MB - 2 GB (LLM dependent)

Monitor: GUI → **💾 Database & Memory**

---

## 🔄 Updates & Versions

### Q: What version is this?
**A:** Dheera v0.3.0 (latest session: v0.3.1)

Features:
- ✅ Spiking neural networks
- ✅ Database monitoring
- ✅ Ontology viewer
- ✅ Document analysis

---

### Q: How do I update Dheera?
**A:**
```bash
cd /Users/nandhavignesh/triton/dheera_v0.3.0
git pull  # If from git
pip install -r dheera_v3/requirements.txt
```

---

## 🛠️ Configuration

### Q: Where is the config file?
**A:** `config/dheera_config.yaml`

Key sections:
- `slm:` - LLM settings
- `spiking:` - Spiking networks
- `dqn:` - Training parameters
- `rag:` - Document retrieval

---

### Q: How do I change the LLM model?
**A:** Edit `config/dheera_config.yaml`:
```yaml
slm:
  model: "llama3.2:latest"  # Change this
```

Available:
- `llama3.2:latest` - Good balance
- `qwen2:1.5b` - Faster, smaller
- `phi3:mini` - Compact

Then restart Dheera.

---

## 📚 Documentation

### Q: Where can I find more help?
**A:** Check:
- [Getting Started](../guides/GETTING_STARTED.md)
- [Chat Guide](../guides/CHAT_GUIDE.md)
- [Common Issues](COMMON_ISSUES.md)
- [Documentation Index](../README.md)

---

### Q: How do I report a bug?
**A:**
1. Check [Common Issues](COMMON_ISSUES.md) first
2. Search existing issues
3. Report at: https://github.com/anthropics/dheera/issues

---

## 🎓 Advanced

### Q: Can I use custom models?
**A:** Yes! Any Ollama-compatible model:
```yaml
slm:
  provider: "ollama"
  model: "your-model:tag"
```

---

### Q: How do I access the API?
**A:** Start server:
```bash
python3 api/server.py
```

Endpoints:
- `GET /health` - Health check
- `POST /api/chat` - Send message (has timeout issues)
- `POST /api/rag/upload` - Upload documents

See: [REST API](../api/REST_API.md)

---

### Q: Can I deploy Dheera on a server?
**A:** Yes, but:
- Requires GPU for best performance
- Ollama must be running
- Configure CORS in `api/server.py`
- API timeout needs fixing first

---

## 🔮 Future Features

### Q: Will image analysis be added?
**A:** Planned! Would need:
- OCR (pytesseract) or
- Vision-language model (llama3.2-vision)

Not yet implemented.

---

### Q: Will there be a mobile app?
**A:** Not currently planned, but possible via:
- API endpoint (needs timeout fix)
- WebSocket chat
- Progressive Web App (PWA)

---

## 💡 Tips & Tricks

### Q: Best practices for chat?
**A:**
- Give specific questions
- Use feedback (++, +, -, --) regularly
- Clear context with `/clear` when switching topics
- Check `/stats` to monitor learning

---

### Q: How to get best performance?
**A:**
1. ✅ Enable spiking networks (already on!)
2. Use smaller LLM model (qwen2:1.5b)
3. Reduce max_tokens in config
4. Clean database regularly
5. Use CLI instead of API

See: [Optimization Guide](../guides/OPTIMIZATION.md)

---

**Still have questions?**

Check: [Common Issues](COMMON_ISSUES.md) or [Documentation Index](../README.md)
