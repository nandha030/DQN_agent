#!/bin/bash
# ============================================
# Dheera v0.3.0 - Folder Structure Setup
# Run: chmod +x setup_dheera.sh && ./setup_dheera.sh
# ============================================

echo "🧠 Creating Dheera v0.3.0 folder structure..."

# Base directory (change this if needed)
BASE_DIR="dheera_v3"

# Create base directory
mkdir -p $BASE_DIR
cd $BASE_DIR

# Core - Rainbow DQN, State Builder, Action Space
mkdir -p core

# Brain - SLM Interface, Executor, Policy
mkdir -p brain

# Cognitive - Intent, Entities, Dialogue State, Working Memory
mkdir -p cognitive

# RAG - Retriever, Embeddings, Vector Store
mkdir -p rag

# RLHF - Reward Model, Preference Learning
mkdir -p rlhf

# Memory - SQLite Store, Replay Buffer, Episodic Memory
mkdir -p memory

# Connectors - Web Search, Tools, Chat Interface
mkdir -p connectors/tools

# Database - Schema, Migrations
mkdir -p database/migrations

# ChromaDB storage
mkdir -p chroma_db

# Config files
mkdir -p config

# Model checkpoints
mkdir -p checkpoints

# Logs
mkdir -p logs

# Tests
mkdir -p tests

# Create __init__.py files for Python packages
touch core/__init__.py
touch brain/__init__.py
touch cognitive/__init__.py
touch rag/__init__.py
touch rlhf/__init__.py
touch memory/__init__.py
touch connectors/__init__.py
touch connectors/tools/__init__.py
touch tests/__init__.py

# Create placeholder files
touch core/rainbow_dqn.py
touch core/state_builder.py
touch core/action_space.py
touch core/curiosity_rnd.py

touch brain/slm_interface.py
touch brain/executor.py
touch brain/policy.py

touch cognitive/intent_classifier.py
touch cognitive/entity_extractor.py
touch cognitive/dialogue_state.py
touch cognitive/working_memory.py
touch cognitive/reasoning.py

touch rag/retriever.py
touch rag/embeddings.py
touch rag/vector_store.py

touch rlhf/reward_model.py
touch rlhf/preference_learner.py
touch rlhf/feedback_collector.py

touch memory/sqlite_store.py
touch memory/replay_buffer.py
touch memory/episodic_memory.py

touch connectors/web_search.py
touch connectors/tool_registry.py
touch connectors/chat_interface.py
touch connectors/tools/calculator.py
touch connectors/tools/python_executor.py

touch database/schema.sql
touch database/db_manager.py

touch config/dheera_config.yaml
touch config/identity.yaml

touch dheera.py
touch run_chat.py
touch requirements.txt

echo ""
echo "✅ Folder structure created!"
echo ""
echo "📁 Structure:"
find . -type f -name "*.py" -o -name "*.yaml" -o -name "*.sql" | head -40
echo ""
echo "📍 Location: $(pwd)"
echo ""
echo "Next steps:"
echo "  cd $BASE_DIR"
echo "  python -m venv venv"
echo "  source venv/bin/activate"
echo "  pip install -r requirements.txt"
