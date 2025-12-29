#!/bin/bash
# setup_dheera.sh
# Dheera v0.3.0 - Complete Setup Script

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║      धीर  DHEERA v0.3.0 - Setup                       ║"
echo "║      Courageous • Wise • Patient                      ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
success() { echo -e "${GREEN}✓ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
error() { echo -e "${RED}✗ $1${NC}"; }

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
major=$(echo $python_version | cut -d'.' -f1)
minor=$(echo $python_version | cut -d'.' -f2)

if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
    success "Python $python_version found"
else
    error "Python 3.9+ required (found $python_version)"
    exit 1
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    success "Virtual environment created"
else
    success "Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate
success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q
success "pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    success "Dependencies installed"
else
    # Install core dependencies
    pip install -q \
        torch>=2.0.0 \
        numpy>=1.24.0 \
        pyyaml>=6.0 \
        requests>=2.28.0 \
        sentence-transformers>=2.2.0
    
    # Optional dependencies
    pip install -q \
        chromadb>=0.3.0 \
        duckduckgo-search>=3.0.0 \
        rich>=13.0.0 \
        tqdm>=4.65.0 \
        || warning "Some optional dependencies failed"
    
    success "Core dependencies installed"
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p chroma_db
mkdir -p config
success "Directories created"

# Initialize database
echo ""
echo "Initializing database..."
if [ -f "database/schema.sql" ]; then
    if [ ! -f "dheera.db" ]; then
        sqlite3 dheera.db < database/schema.sql
        success "Database initialized"
    else
        success "Database exists"
    fi
else
    warning "Schema file not found, database will be created on first run"
fi

# Check Ollama
echo ""
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    success "Ollama found"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        success "Ollama is running"
        
        # Check for phi3:mini
        if ollama list | grep -q "phi3:mini"; then
            success "phi3:mini model found"
        else
            echo "Pulling phi3:mini model..."
            ollama pull phi3:mini
            success "phi3:mini model pulled"
        fi
    else
        warning "Ollama not running. Start with: ollama serve"
    fi
else
    warning "Ollama not found. Install from: https://ollama.ai"
    echo "  Dheera will use echo mode for testing without Ollama."
fi

# Run tests
echo ""
echo "Running basic tests..."

# Test imports
python3 -c "
import sys
sys.path.insert(0, '.')

print('Testing imports...')

try:
    from core.rainbow_dqn import RainbowDQNAgent
    print('  ✓ Rainbow DQN')
except ImportError as e:
    print(f'  ✗ Rainbow DQN: {e}')

try:
    from core.curiosity_rnd import CuriosityModule
    print('  ✓ Curiosity RND')
except ImportError as e:
    print(f'  ✗ Curiosity RND: {e}')

try:
    from cognitive.intent_classifier import IntentClassifier
    print('  ✓ Intent Classifier')
except ImportError as e:
    print(f'  ✗ Intent Classifier: {e}')

try:
    from rag.retriever import RAGRetriever
    print('  ✓ RAG Retriever')
except ImportError as e:
    print(f'  ✗ RAG Retriever: {e}')

try:
    from rlhf.feedback_collector import FeedbackCollector
    print('  ✓ RLHF')
except ImportError as e:
    print(f'  ✗ RLHF: {e}')

try:
    from dheera import Dheera
    print('  ✓ Main Orchestrator')
except ImportError as e:
    print(f'  ✗ Main Orchestrator: {e}')

print('Import tests completed!')
"

success "Tests completed"

# Print summary
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║              Setup Complete!                          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "To get started:"
echo ""
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start Ollama (if not running):"
echo "     ollama serve"
echo ""
echo "  3. Run Dheera:"
echo "     python run_chat.py"
echo ""
echo "  4. Or run demo:"
echo "     python run_demo.py --full"
echo ""
echo "  5. Or run training:"
echo "     python training_session.py --episodes 10"
echo ""
echo "For more options, use --help with any script."
echo ""
