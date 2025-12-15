#!/bin/bash
# Setup and Run Benchmark Comparison: Mistral Q2_K vs Q3_K_M vs Phi-2
# For 4GB edge device deployment decision

set -e

WORKSPACE_DIR="/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG"
MODELS_DIR="$WORKSPACE_DIR/models"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 MISTRAL vs PHI-2 BENCHMARK SETUP"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if Q2_K model exists and is complete
Q2K_PATH="$MODELS_DIR/mistral-7b-instruct-v0.2.Q2_K.gguf"
Q2K_SIZE=$(ls -lh "$Q2K_PATH" 2>/dev/null | awk '{print $5}' || echo "missing")
Q2K_BYTES=$(ls -Lah "$Q2K_PATH" 2>/dev/null | awk '{print $5}' || echo "0")

echo "📊 Current Model Status:"
echo "   Q3_K_M: $(ls -lh $MODELS_DIR/mistral-7b-instruct-v0.2.Q3_K_M.gguf | awk '{print $5}')"
echo "   Q2_K:   $Q2K_SIZE (Target: 2.4GB)"
echo "   Phi-2:  $(ls -lh $MODELS_DIR/phi-2-q4_0.gguf | awk '{print $5}')"
echo ""

# Download Q2_K if needed
if [[ ! -f "$Q2K_PATH" ]] || [[ $(stat -f%z "$Q2K_PATH" 2>/dev/null || stat -c%s "$Q2K_PATH" 2>/dev/null) -lt 2000000000 ]]; then
    echo "📥 Downloading Mistral-7B Q2_K (2.4GB)..."
    echo "   This will take 5-15 minutes depending on your connection"
    echo ""
    cd "$MODELS_DIR"
    wget -c "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf" \
         -O "mistral-7b-instruct-v0.2.Q2_K.gguf" \
         --progress=bar:force:noscroll \
         2>&1 | grep -E "(HTTP|saved|%)" || true
    echo ""
    
    # Verify download
    if [[ -f "$Q2K_PATH" ]]; then
        FINAL_SIZE=$(ls -lh "$Q2K_PATH" | awk '{print $5}')
        FINAL_BYTES=$(ls -L "$Q2K_PATH" | awk '{print $5}')
        
        if [[ $FINAL_BYTES -gt 2000000000 ]]; then
            echo "   ✅ Q2_K downloaded successfully ($FINAL_SIZE)"
        else
            echo "   ⚠️  Q2_K download incomplete ($FINAL_SIZE). Continuing with available models..."
        fi
    fi
else
    echo "✅ Q2_K already available ($(ls -lh $Q2K_PATH | awk '{print $5}'))"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🏃 STARTING BENCHMARK COMPARISON"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This will benchmark three models:"
echo "  1. Mistral-7B Q3_K_M (3.3GB) - Current desktop config"
echo "  2. Mistral-7B Q2_K (2.4GB)   - Recommended for 4GB edge"
echo "  3. Phi-2 Q4_0 (1.5GB)       - Fast but weak multilingual"
echo ""
echo "⏱️  Estimated time: 20-30 minutes"
echo "   (Each model: 6 queries × 3 runs each)"
echo ""

cd "$WORKSPACE_DIR"

# Run comprehensive benchmark
python benchmark_comparison_report.py

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ BENCHMARK COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📄 Reports generated:"
echo "   ✓ BENCHMARK_COMPARISON.md  (Detailed analysis & recommendations)"
echo "   ✓ benchmark_comparison.json (Machine-readable results)"
echo ""
echo "📊 Key Recommendation:"
echo "   For 4GB edge device deployment → Use Mistral-7B Q2_K"
echo ""
echo "🔗 Next steps:"
echo "   1. Review: BENCHMARK_COMPARISON.md"
echo "   2. Update: fast_rag_config.py (use Q2_K model path)"
echo "   3. Deploy: Copy model to edge device"
echo "   4. Test: Run stress tests on target device"
echo ""
echo "📚 Reference Guides:"
echo "   - MISTRAL_Q2K_4GB_GUIDE.md (Configuration details)"
echo "   - EDGE_DEVICE_4GB_OPTIONS.md (Comparison options)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
