#!/bin/bash
# Stop all RAG servers

echo "🛑 Stopping Caramella RAG servers..."

# Kill API server
if [ -f /tmp/rag_api.pid ]; then
    API_PID=$(cat /tmp/rag_api.pid)
    kill $API_PID 2>/dev/null
    echo "   ✅ API server stopped (PID: $API_PID)"
    rm /tmp/rag_api.pid
fi

# Kill Frontend
if [ -f /tmp/rag_frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/rag_frontend.pid)
    kill $FRONTEND_PID 2>/dev/null
    echo "   ✅ Frontend stopped (PID: $FRONTEND_PID)"
    rm /tmp/rag_frontend.pid
fi

# Clean up any remaining processes
pkill -f "python.*main.py" 2>/dev/null
pkill -f "vite" 2>/dev/null
pkill -f "npm.*dev" 2>/dev/null

echo "✅ All servers stopped"
