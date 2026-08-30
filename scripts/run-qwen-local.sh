#!/usr/bin/env bash
# Run Qwen3-8B-abliterated via llama.cpp with CUDA offload (RTX 4050 6GB)
MODEL="/home/mevlec/Data/arc/Qwen3-8B-abliterated-q4_k_m.gguf"
PORT=8080
if [[ ! -f "$MODEL" ]]; then
  ALT="/home/mevlec/Downloads/Qwen3-8B-abliterated-q4_k_m.gguf"
  [[ -f "$ALT" ]] && MODEL="$ALT"
fi
echo "Starting llama-server: $MODEL on http://127.0.0.1:$PORT/v1 (model: Qwen3-8B-abliterated)"
echo "VRAM offload: auto (fits 4.9G in 6GB), ctx 8192"
# Ensure nvidia libs in path (from venv)
export LD_LIBRARY_PATH="/home/mevlec/Data/arc/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/mevlec/Data/arc/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
exec /home/mevlec/llama.cpp/build/bin/llama-server \
  --model "$MODEL" \
  --alias Qwen3-8B-abliterated \
  --host 127.0.0.1 --port $PORT \
  --n-gpu-layers auto --ctx-size 8192 --threads 8 \
  --flash-attn auto
