#!/bin/bash
# Start the FastAPI server with GPU support

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Caramella RAG API Server..."
echo "📁 Project directory: $PROJECT_DIR"
echo "🔧 Using GPU acceleration with CUDA libraries"
echo ""

# Export CUDA library paths
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu

# Run the API server
cd "$SCRIPT_DIR"
"$PROJECT_DIR/run_with_gpu.sh" python main.py

# Alternative: Use uvicorn directly for production with workers
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
