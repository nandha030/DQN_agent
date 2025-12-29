#!/bin/bash
# Start Dheera Professional GUI

echo "🚀 Starting Dheera Professional GUI..."
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Backend not running. Starting backend..."
    cd "$(dirname "$0")"
    python3 api/server.py > /tmp/dheera_api.log 2>&1 &
    sleep 3
    echo "✅ Backend started"
fi

# Check if virtual environment exists
if [ -d "/Users/nandhavignesh/triton/triton-client-env" ]; then
    echo "✅ Using virtual environment"
    source /Users/nandhavignesh/triton/triton-client-env/bin/activate
fi

# Start Streamlit
echo "🎨 Launching Professional GUI..."
echo ""
echo "📍 Open in browser: http://localhost:8501"
echo ""

streamlit run gui_professional.py
