#!/bin/bash
# Start both API server and Frontend for mobile testing

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  🍬 Starting Caramella RAG System (API + UI)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Get machine IP address
echo "🔍 Detecting network addresses..."
IP_ADDR=$(hostname -I | awk '{print $1}')
LOCALHOST="127.0.0.1"

echo ""
echo "📍 Access URLs:"
echo "   Desktop (Local):   http://localhost:5173"
echo "   Desktop (Network): http://$IP_ADDR:5173"
echo "   Mobile (WiFi):     http://$IP_ADDR:5173"
echo ""
echo "   API Docs:          http://localhost:8000/api/docs"
echo "   API Network:       http://$IP_ADDR:8000/api/docs"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Start API server in background
echo "🚀 Starting FastAPI server..."
cd /mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/api
python main.py > api.log 2>&1 &
API_PID=$!
echo "   ✅ API server started (PID: $API_PID)"
echo "   📝 Logs: api/api.log"

# Wait for API to be ready
echo ""
echo "⏳ Waiting for API to initialize (loading Mistral-7B model)..."
sleep 3
for i in {1..20}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "   ✅ API is ready!"
        break
    fi
    echo "   ⏳ Still loading... ($i/20)"
    sleep 2
done

# Start frontend
echo ""
echo "🎨 Starting React frontend..."
cd /mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/frontend
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   ✅ Frontend started (PID: $FRONTEND_PID)"
echo "   📝 Logs: frontend/frontend.log"

sleep 3

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  ✅ ALL SERVICES RUNNING"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📱 MOBILE TESTING INSTRUCTIONS:"
echo ""
echo "   1. Make sure your phone is on the SAME WiFi network as this computer"
echo "   2. Open your phone's browser (Safari/Chrome)"
echo "   3. Navigate to: http://$IP_ADDR:5173"
echo "   4. Try sample Korean/English queries"
echo ""
echo "💻 DESKTOP ACCESS:"
echo "   • Open browser: http://localhost:5173"
echo "   • API docs: http://localhost:8000/api/docs"
echo ""
echo "🛑 TO STOP SERVERS:"
echo "   • Press Ctrl+C or run: ./stop_servers.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Save PIDs for cleanup
echo $API_PID > /tmp/rag_api.pid
echo $FRONTEND_PID > /tmp/rag_frontend.pid

# Keep script running and show logs
echo "📊 Live Logs (Ctrl+C to stop):"
echo ""
tail -f /mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/api/api.log
