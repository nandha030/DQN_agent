# 📘 Dheera v0.3.0 - Complete User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Core Features](#core-features)
3. [GUI Interface](#gui-interface)
4. [Search & AI Page](#search--ai-page)
5. [Model Management](#model-management)
6. [Advanced Features](#advanced-features)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

# Quick Start

## 🚀 Get Started in 3 Steps

### 1. Start the Backend
```bash
cd /Users/nandhavignesh/triton/dheera_v0.3.0/dheera_v3
python3 api/server.py
```

### 2. Start the GUI
```bash
streamlit run gui_professional.py --server.port 8501
```

### 3. Access
- **GUI**: http://localhost:8501
- **API**: http://localhost:8000

---

# Core Features

## 🧠 Dheera Architecture

### **Brain-Inspired AI with Hot-Swappable LLMs**

Dheera combines multiple cutting-edge AI techniques:

#### 1. **Rainbow DQN (Deep Q-Network)**
- **What it does**: Learns optimal conversation strategies through reinforcement learning
- **Components**:
  - Double DQN: Reduces overestimation bias
  - Dueling Networks: Separates value and advantage estimation
  - Prioritized Experience Replay: Learns from important experiences
  - Multi-step Returns: Better long-term planning
  - Noisy Networks: Exploration through parameter noise

#### 2. **Curiosity-Driven Exploration**
- **ICM (Intrinsic Curiosity Module)**: Encourages exploring novel conversation paths
- **Reward Bonuses**: For discovering new interaction patterns
- **Adaptive Learning**: Balances exploration vs exploitation

#### 3. **RLHF (Reinforcement Learning from Human Feedback)**
- **User Ratings**: 👍 Thumbs up/down on responses
- **Reward Modeling**: Learns what makes good responses
- **Preference Learning**: Aligns with your preferences over time

#### 4. **RAG (Retrieval-Augmented Generation)**
- **Document Ingestion**: Add your own knowledge base
- **Semantic Search**: Finds relevant context using embeddings
- **ChromaDB Backend**: Efficient vector storage
- **Context Integration**: Enriches responses with retrieved info

#### 5. **Spiking Neural Networks (Experimental)**
- **Brain-like Processing**: Mimics biological neurons
- **Temporal Dynamics**: Time-based spike patterns
- **Energy Efficient**: Sparse activation

---

# GUI Interface

## 💬 Chat Page

### **Main Features**
- Real-time conversation with AI
- Session management (create, rename, delete)
- Chat history with timestamps
- Export conversations
- Rate responses (👍/👎) for RLHF

### **How to Use**
1. Navigate to "💬 Chat" page
2. Type your message in the input box
3. Press Enter or click Send
4. Rate responses with thumbs up/down

---

## 🔍 Search & AI Page

### **Professional Google-Style Interface**

#### **Tab 1: Web Search**
- **Free DuckDuckGo search** (no API key needed)
- AI-powered summarization
- Source citations with links
- Adjustable result count (1-10)

**Example:**
```
Query: "latest Python 3.13 features"
→ Searches web (466ms)
→ AI summarizes findings
→ Shows official sources
→ Time: 2-3 seconds total
```

#### **Tab 2: Multi-Model Comparison**
- Query 2-4 models simultaneously
- Parallel execution (not sequential)
- Side-by-side comparison
- Latency tracking
- Consensus generation

**Example:**
```
Query: "Explain quantum computing"
Models: qwen2, phi3, llama3.2
→ All queried in parallel
→ Compare quality & speed
→ Get unified summary
```

#### **Tab 3: Quick Answer**
- Instant response from active model
- Model metadata display
- Fast, simple interface

---

## 🔧 Models Page

### **Tab 1: Available Models**
- View all configured models
- See active/inactive status
- Check availability
- View usage statistics

### **Tab 2: Add API**
Add external API providers:

**Free APIs (No Credit Card):**
1. **Groq** (Ultra-fast, 300+ tokens/sec)
   - URL: https://console.groq.com
   - Model: llama-3.3-70b-versatile
   - Free: 30 requests/minute

2. **Google Gemini** (Latest Google AI)
   - URL: https://ai.google.dev
   - Model: gemini-2.0-flash-exp
   - Free: 15 requests/minute

3. **Together AI** ($25 free credits)
   - URL: https://api.together.xyz
   - Model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
   - Credits: $25 for new users

**Setup:**
1. Go to "➕ Add API" tab
2. Select provider type: Custom (OpenAI-compatible)
3. Fill in details:
   - Provider Name: `groq_llama`
   - Model: `llama-3.3-70b-versatile`
   - Base URL: `https://api.groq.com/openai/v1`
   - API Key: (from provider website)
4. Click "Add Provider"
5. Activate in "Available Models" tab

### **Tab 3: Test Models**
- Send test queries to any model
- Compare response quality
- Check latency

### **Tab 4: Compare Models**
- Side-by-side comparison
- Multiple models at once
- Performance metrics

---

## 📊 Analytics Page

### **Session Statistics**
- Total messages sent
- Average message length
- Session duration
- Active time tracking

### **DQN Training Metrics**
- Episodes completed
- Average reward
- Epsilon (exploration rate)
- Loss values
- Steps per episode

### **Model Performance**
- Requests per model
- Total tokens used
- Average latency
- Error counts

---

## ⚙️ Settings Page

### **Theme Settings**
- Auto-detect system theme
- Manual dark/light toggle
- Responsive design

### **Model Configuration**
- Switch active model
- View current settings
- Hot-swap without restart

### **Advanced Settings**
- DQN training parameters
- RAG configuration
- RLHF reward weights
- Curiosity parameters

---

# Advanced Features

## 📚 RAG (Knowledge Base)

### **Adding Documents**
1. Go to Settings → RAG Configuration
2. Upload PDF, TXT, or DOCX files
3. Documents are chunked and embedded
4. Stored in ChromaDB

### **How It Works**
- User query triggers semantic search
- Top-K relevant chunks retrieved
- Chunks added to LLM context
- Response enriched with your knowledge

### **Configuration**
```yaml
rag:
  enabled: true
  default_n_results: 3
  min_score: 0.5
  max_context_tokens: 300
  max_context_chars: 1200
```

---

## 🎯 RLHF (Human Feedback)

### **Rating Responses**
- Click 👍 for good responses
- Click 👎 for bad responses
- System learns your preferences

### **How It Works**
1. Your ratings train a reward model
2. Model predicts what you'll like
3. DQN optimizes for higher rewards
4. Responses improve over time

### **View Training**
- Go to Analytics → RLHF Metrics
- See reward trends
- Check preference alignment

---

## 🌊 DQN Training

### **Real-Time Learning**
- Learns optimal conversation strategies
- Balances exploration vs exploitation
- Adapts to your interaction patterns

### **Key Metrics**
- **Reward**: Higher is better conversation quality
- **Epsilon**: Exploration rate (decreases over time)
- **Loss**: Training stability (should decrease)

### **Training Configuration**
```yaml
dqn:
  gamma: 0.99          # Discount factor
  learning_rate: 0.0001
  batch_size: 32
  train_every: 10      # Steps between training
  target_update: 100   # Steps to update target network
```

---

## 🔥 Curiosity-Driven Learning

### **Intrinsic Motivation**
- Explores novel conversation topics
- Discovers new interaction patterns
- Prevents getting stuck in local optima

### **Configuration**
```yaml
curiosity:
  enabled: true
  beta: 0.2           # Curiosity weight
  eta: 0.01           # Forward model learning rate
```

---

# Performance Optimization

## ⚡ Latency Improvements

### **Current Performance**
- Simple queries: 2-5s
- Complex queries: 5-10s
- Web search: 2-3s total

### **Optimizations Applied**
1. **SLM Timeout**: 60s → 15s
2. **Max Tokens**: 512 → 256
3. **DQN Batch Size**: 64 → 32
4. **RAG Results**: 5 → 3
5. **Smart RAG Skipping**: Skip for greetings

### **Model Selection**
- **Fastest**: Groq (0.5-1s)
- **Balanced**: Gemini (1-2s)
- **Privacy**: Local Ollama (2-5s)

### **Hardware Impact**
- **GPU**: 2-4x faster inference
- **RAM**: 16GB+ recommended for large models
- **SSD**: Faster model loading

---

## 🎨 Model Recommendations

### **For Speed**
1. Groq (ultra-fast, free)
2. Gemini Flash (fast, free)
3. Qwen2:1.5b (local, small)

### **For Quality**
1. Claude via OpenRouter ($5 credits)
2. GPT-4 via OpenRouter
3. Llama 3.2 70B (local, large)

### **For Cost**
1. Local Ollama (free, unlimited)
2. Groq (free tier)
3. Gemini (free tier)

---

# Model Management

## 🔄 Hot-Swapping Models

### **Switch Models Instantly**
1. Go to "🔧 Models" page
2. Click "Activate" on any model
3. No backend restart needed
4. Immediate switch

### **Multi-Model Setup**
Add multiple providers:
- **Primary**: Groq (fast)
- **Backup**: Local Ollama (always works)
- **Fallback**: Gemini (rate limit backup)

---

## 📥 Adding Local Models

### **Using Ollama**
```bash
# List available models
ollama list

# Pull new model
ollama pull llama3.2:latest

# Models auto-detected by Dheera
# Refresh GUI to see them
```

---

## 🌐 Web Search Providers

### **DuckDuckGo (Free, Default)**
- No API key needed
- Unlimited searches
- Privacy-focused
- Instant answers

### **SerpAPI (100 free/month)**
1. Sign up: https://serpapi.com
2. Get API key
3. Add to web_search_tool.py
4. Higher quality results

### **Brave Search (Cheap)**
1. Sign up: https://brave.com/search/api
2. Get API key
3. Independent index
4. $0.003 per search

---

# Troubleshooting

## ❌ Common Issues

### **"Backend not running"**
```bash
# Check if running
curl http://localhost:8000

# Start backend
cd dheera_v3
python3 api/server.py
```

### **"No models available"**
```bash
# Check Ollama
ollama list

# Pull a model
ollama pull qwen2:1.5b

# Restart backend
```

### **"Web search failed"**
- Check internet connection
- DuckDuckGo doesn't need API key
- Try different provider

### **"Streamlit won't start"**
```bash
# Kill old processes
pkill -9 streamlit

# Restart
streamlit run gui_professional.py --server.port 8501
```

### **"High latency"**
1. Check [Performance Optimization](#performance-optimization)
2. Reduce max_tokens in config
3. Use faster model (Groq)
4. Skip RAG for simple queries

---

## 🐛 Debug Mode

### **Enable Detailed Logs**
```bash
python3 run_chat.py --debug
```

### **Check Logs**
```bash
# Backend logs
tail -f api/server.log

# DQN training
tail -f logs/training.log
```

---

# System Requirements

## 💻 Minimum
- **OS**: macOS, Linux, Windows
- **Python**: 3.9+
- **RAM**: 8GB
- **Storage**: 10GB

## 🚀 Recommended
- **OS**: macOS/Linux
- **Python**: 3.11+
- **RAM**: 16GB+
- **GPU**: NVIDIA (for fast inference)
- **Storage**: 50GB (for multiple models)

---

# Configuration Reference

## 📄 config/dheera_config.yaml

### **Full Configuration**
```yaml
# SLM (Small Language Model)
slm:
  enabled: true
  provider: "ollama"
  model: "qwen2:1.5b"
  timeout: 15
  max_tokens: 256
  temperature: 0.7

# DQN (Deep Q-Network)
dqn:
  state_size: 768
  action_size: 10
  hidden_size: 256
  gamma: 0.99
  learning_rate: 0.0001
  batch_size: 32
  train_every: 10
  target_update: 100
  min_experiences: 50

# RAG (Retrieval-Augmented Generation)
rag:
  enabled: true
  persist_directory: "chroma_db"
  collection_name: "dheera_knowledge"
  embedding_model: "mxbai-embed-large:latest"
  default_n_results: 3
  min_score: 0.5
  max_context_tokens: 300
  max_context_chars: 1200

# RLHF (Reinforcement Learning from Human Feedback)
rlhf:
  enabled: true
  reward_scale: 1.0
  positive_reward: 1.0
  negative_reward: -1.0

# Curiosity
curiosity:
  enabled: true
  beta: 0.2
  eta: 0.01
  feature_dim: 128

# Spiking Networks (Experimental)
spiking:
  enabled: false
  num_neurons: 1000
  threshold: 1.0
  decay: 0.9
```

---

# API Reference

## 🔌 Backend Endpoints

### **Chat**
```bash
POST /api/chat
{
  "message": "Hello",
  "session_id": "optional"
}

Response:
{
  "response": "Hi there!",
  "metadata": {
    "model": "qwen2:1.5b",
    "tokens_used": 45,
    "latency_ms": 850
  }
}
```

### **Switch Model**
```bash
POST /api/llm/switch
{
  "name": "groq_llama"
}
```

### **List Models**
```bash
GET /api/llm/providers

Response:
{
  "providers": [
    {
      "name": "ollama_qwen2_1_5b",
      "active": true,
      "available": true
    }
  ]
}
```

### **Stats**
```bash
GET /api/stats

Response:
{
  "dheera": {
    "dqn": {
      "episodes": 150,
      "avg_reward": 0.85
    }
  }
}
```

---

# Development

## 🛠️ File Structure

```
dheera_v3/
├── dheera.py                    # Core DQN agent
├── api/
│   ├── server.py               # FastAPI backend
│   └── llm_router.py           # Model management
├── utils/
│   ├── rag_engine.py           # RAG implementation
│   ├── system_profiler.py      # Auto-optimization
│   └── goal_evaluator.py       # Goal tracking
├── config/
│   └── dheera_config.yaml      # Configuration
├── connectors/
│   ├── tools/
│   │   ├── web_search_tool.py  # Web search
│   │   └── multi_model_agent.py # Multi-model
│   └── chat_interface.py       # Chat abstraction
├── gui_professional.py          # Streamlit GUI
└── chroma_db/                   # RAG database
```

---

# Credits & License

## 👨‍💻 Dheera v0.3.0

**Brain-Inspired AI with Hot-Swappable LLM Backends**

### **Technologies**
- Rainbow DQN (DeepMind)
- RLHF (OpenAI)
- RAG (Facebook AI)
- Spiking Neural Networks
- Curiosity-Driven Learning

### **Dependencies**
- PyTorch (Deep Learning)
- FastAPI (Backend)
- Streamlit (GUI)
- ChromaDB (Vector Storage)
- Ollama (Local LLMs)

---

# Quick Reference

## 🚀 Common Commands

```bash
# Start backend
python3 api/server.py

# Start GUI
streamlit run gui_professional.py

# Debug mode
python3 run_chat.py --debug

# Pull new model
ollama pull llama3.2

# Check health
curl http://localhost:8000

# View logs
tail -f logs/dheera.log
```

## 🔗 Useful Links

- **Groq API**: https://console.groq.com
- **Gemini API**: https://ai.google.dev
- **Ollama**: https://ollama.ai
- **OpenRouter**: https://openrouter.ai
- **Together AI**: https://api.together.xyz

---

## 📞 Support

For issues, check:
1. [Troubleshooting](#troubleshooting)
2. Debug logs: `python3 run_chat.py --debug`
3. Backend health: `curl http://localhost:8000`

---

**End of Complete Guide**

🎉 Enjoy using Dheera v0.3.0!
