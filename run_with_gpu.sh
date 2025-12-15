#!/bin/bash
# Wrapper to run Python with correct library paths for GPU acceleration
# Usage: ./run_with_gpu.sh python script.py [args...]

# Add CUDA libs to existing path (conda libs remain first for PyTorch)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.1/lib64

# Execute the command with all arguments
exec "$@"
