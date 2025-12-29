#!/bin/bash
# Quick start script for Dheera GUI

echo "🚀 Starting Dheera GUI..."
echo ""

# Check if requirements are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install fastapi uvicorn websockets streamlit plotly
    echo ""
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "⚠️  Warning: Ollama is not running"
    echo "   Start it with: ollama serve"
    echo ""
fi

# Start backend in background
echo "1️⃣  Starting FastAPI backend..."
python3 api/server.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start Streamlit GUI
echo ""
echo "2️⃣  Starting Streamlit GUI..."
echo "   GUI will open at: http://localhost:8501"
echo "   API docs at: http://localhost:8000/docs"
echo ""
streamlit run gui_streamlit.py

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
