#!/bin/bash
# Check Fast RAG Service Status

echo "Checking Fast RAG Service..."
echo ""

# Check if process is running
PID=$(ps aux | grep "fast_rag_service" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "❌ Service is NOT running"
    echo ""
    echo "Start it with:"
    echo "  python fast_rag_service.py --profile balanced"
    exit 1
fi

echo "✅ Service process running (PID: $PID)"
echo ""

# Check if service responds
echo "Testing HTTP endpoint..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")

if [ "$HTTP_STATUS" == "200" ]; then
    echo "✅ Service is READY and responding!"
    echo ""
    echo "📊 Health check:"
    curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
    echo ""
    echo ""
    echo "🌐 Open this in your browser:"
    echo "   file:///mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/web_demo.html"
    echo ""
    echo "Or test with:"
    echo "  curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\": \"Test\"}'"
else
    echo "⏳ Service is LOADING (still initializing models)..."
    echo ""
    echo "Latest logs:"
    tail -5 /mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/service.log
    echo ""
    echo "💡 Model loading typically takes 30-60 seconds"
    echo "   Run this script again in a few seconds..."
fi
