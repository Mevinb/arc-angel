#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

MODEL_PATH="/home/mevlec/Downloads/dolphin3.0-llama3.1-8b-q4_k_m.gguf"
LLAMA_SERVER="/home/mevlec/Downloads/llama.cpp/build/bin/llama-server"

echo "=========================================================="
echo "      GROK // FULL GPU RUNNER (RTX 4050 6GB CUDA)         "
echo "=========================================================="
echo "[+] Model: $(basename "$MODEL_PATH")"
echo "[+] Binary: $LLAMA_SERVER"
echo "[+] GPU Layers: 33 (100% Offload to RTX 4050)"
echo "[+] Flash Attention: ON"
echo "[+] Context Window: 4096 tokens"
echo "=========================================================="
echo "[*] Launching llama-server on http://127.0.0.1:8080 ..."
echo "[*] Press Ctrl+C to stop."
echo ""

exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8080 \
    -c 4096 \
    -ngl 33 \
    -t 8 \
    -fa on \
    --split-mode none \
    --alias dolphin-grok
