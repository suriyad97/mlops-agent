#!/bin/bash
# ─────────────────────────────────────────────────────────────
# MLOps Agent — Start both backend and frontend
# Run: bash run.sh
# ─────────────────────────────────────────────────────────────

echo "⚡ Starting MLOps Agent..."

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded .env"
else
    echo "✗ .env file not found — copy .env.example to .env and fill in your values"
    exit 1
fi

# Start FastAPI backend in background
echo "▶ Starting FastAPI backend on http://localhost:8000 ..."
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start Streamlit frontend
echo "▶ Starting Streamlit UI on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501

# Cleanup backend when Streamlit exits
kill $BACKEND_PID
echo "✓ Stopped."
