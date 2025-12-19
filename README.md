# धीर Dheera v0.3.0

> **धीर** (Sanskrit): Courageous, Wise, Patient

A hybrid AI agent combining Reinforcement Learning with Small Language Models.

## 🌟 Features

### Core Architecture
- **Rainbow DQN** - All 6 improvements for efficient action selection
  - Double DQN (reduced overestimation)
  - Dueling Networks (value/advantage separation)
  - Noisy Networks (learned exploration)
  - Prioritized Experience Replay (focus on important transitions)
  - N-step Returns (better credit assignment)
  - Distributional RL (C51 - value distribution learning)

- **RND Curiosity** - Intrinsic motivation for exploration
  - Random Network Distillation
  - Novelty-based rewards
  - Prevents local optima

- **Cognitive Layer** - Intent understanding and reasoning
  - Intent Classification (20+ intents)
  - Entity Extraction (14+ entity types)
  - Dialogue State Tracking
  - Working Memory
  - Reasoning Engine

- **RAG System** - Long-term memory
  - ChromaDB vector database
  - Sentence transformer embeddings
  - Multi-source retrieval

- **RLHF** - Learning from human feedback
  - Reward model training
  - Preference learning (Bradley-Terry)
  - Feedback collection (++, +, -, --)

- **Memory System** - Episodic and replay buffers
  - Episodic memory for conversations
  - Prioritized replay buffer
  - SQLite persistence

## 📁 Project Structure

```
dheera_v3/
├── dheera.py              # Main orchestrator
├── run_chat.py            # Interactive chat interface
├── run_demo.py            # Demo script
├── training_session.py    # Automated training
├── setup_dheera.sh        # Setup script
├── requirements.txt       # Dependencies
│
├── config/
│   ├── dheera_config.yaml # Main configuration
│   └── identity.yaml      # Agent identity
│
├── core/
│   ├── rainbow_dqn.py     # Rainbow DQN agent
│   ├── curiosity_rnd.py   # RND curiosity module
│   ├── state_builder.py   # 64-dim state construction
│   └── action_space.py    # 8-action space definition
│
├── brain/
│   ├── slm_interface.py   # Ollama/SLM interface
│   ├── executor.py        # Action executor
│   └── policy.py          # Safety guardrails
│
├── cognitive/
│   ├── intent_classifier.py   # Intent classification
│   ├── entity_extractor.py    # Entity extraction
│   ├── dialogue_state.py      # Conversation tracking
│   ├── working_memory.py      # Short-term memory
│   └── reasoning.py           # Reasoning engine
│
├── rag/
│   ├── embeddings.py      # Text embeddings
│   ├── vector_store.py    # Vector database
│   └── retriever.py       # RAG retrieval
│
├── rlhf/
│   ├── reward_model.py        # Reward prediction
│   ├── preference_learner.py  # Preference learning
│   └── feedback_collector.py  # Feedback collection
│
├── memory/
│   ├── episodic_memory.py # Conversation episodes
│   ├── replay_buffer.py   # Experience replay
│   └── sqlite_store.py    # Persistent storage
│
├── connectors/
│   ├── chat_interface.py  # Terminal chat UI
│   ├── tool_registry.py   # Tool management
│   ├── web_search.py      # Web search
│   └── tools/
│       ├── calculator.py      # Math tool
│       └── python_executor.py # Code execution
│
├── database/
│   ├── schema.sql         # SQLite schema
│   └── db_manager.py      # Database operations
│
└── tests/
    └── __init__.py        # Test suite
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai) (for local SLM)

### 2. Setup
```bash
# Clone/download the project
cd dheera_v3

# Run setup
chmod +x setup_dheera.sh
./setup_dheera.sh

# Or manual setup:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Ollama
```bash
# Start Ollama server
ollama serve

# Pull the model (in another terminal)
ollama pull phi3:mini
```

### 4. Run Dheera
```bash
# Activate virtual environment
source venv/bin/activate

# Start chat
python run_chat.py

# Or run demo
python run_demo.py --full

# Or run training
python training_session.py --episodes 10
```

## 💬 Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/stats` | Show statistics |
| `/save` | Save checkpoint |
| `/load PATH` | Load checkpoint |
| `/reset` | Reset conversation |
| `/search Q` | Force web search |
| `/debug` | Toggle debug mode |
| `/quit` | Exit |

## 📊 Feedback System

Provide feedback after any response:

| Feedback | Meaning | Value |
|----------|---------|-------|
| `++` | Very positive | +1.0 |
| `+` | Positive | +0.5 |
| `-` | Negative | -0.5 |
| `--` | Very negative | -1.0 |

Feedback trains the RLHF reward model!

## 🎯 Action Space

| ID | Action | Description |
|----|--------|-------------|
| 0 | DIRECT_RESPONSE | Answer directly |
| 1 | CLARIFY_QUESTION | Ask for clarification |
| 2 | USE_TOOL | Execute a tool |
| 3 | SEARCH_WEB | Search the internet |
| 4 | BREAK_DOWN_TASK | Decompose complex task |
| 5 | REFLECT_AND_REASON | Step-by-step thinking |
| 6 | DEFER_OR_DECLINE | Politely decline |
| 7 | COGNITIVE_PROCESS | Use cognitive layer |

## 🧠 State Space (64 dimensions)

| Range | Features |
|-------|----------|
| 0-15 | Original (sentiment, complexity, etc.) |
| 16-31 | Semantic (compressed embeddings) |
| 32-47 | Cognitive (intent, entities, dialogue) |
| 48-63 | Context (RAG, memory, history) |

## ⚙️ Configuration

Edit `config/dheera_config.yaml`:

```yaml
slm:
  provider: "ollama"
  model: "phi3:mini"
  temperature: 0.7

dqn:
  gamma: 0.99
  lr: 0.0001
  batch_size: 64
  curiosity_coef: 0.1

rag:
  n_results: 5
  min_score: 0.3
```

## 🔧 Development

```bash
# Run tests
python -m pytest tests/

# Run specific component
python -m core.rainbow_dqn
python -m cognitive.intent_classifier
python -m rag.retriever
```

## 📈 Training

```bash
# Basic training
python training_session.py --episodes 50 --turns 15

# With custom feedback rate
python training_session.py --episodes 100 --feedback 0.5

# Load existing checkpoint
python training_session.py --checkpoint checkpoints/model.pt
```

## 🛡️ Safety Features

- **Policy Guard** - Content filtering and safety checks
- **PII Detection** - Automatic redaction
- **Rate Limiting** - Request throttling
- **Input Validation** - Sanitization

## 📝 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Anthropic (Claude) for guidance
- Ollama for local SLM inference
- Sentence-Transformers for embeddings
- ChromaDB for vector storage

---

**Created by Nandha Vignesh**

*धीर - Courageous in learning, Wise in responses, Patient in exploration*
