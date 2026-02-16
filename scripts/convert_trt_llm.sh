#!/bin/bash
# Conversion HuggingFace -> TensorRT-LLM (Jetson)

# --- Parametres a adapter ---
MODEL_DIR="/home/jetson/models/llama-3.2-1b"   # dossier du modele HF
TRTLLM_DIR="/home/jetson/TensorRT-LLM"        # repo TensorRT-LLM
OUT_BASE="/home/jetson/trt_engines/llama-3.2-1b"

CKPT_DIR="${OUT_BASE}/checkpoint"
ENGINE_DIR="${OUT_BASE}/engine"

# --- Creation des dossiers ---
mkdir -p "$CKPT_DIR" "$ENGINE_DIR"

# --- Conversion HF -> Checkpoint TensorRT-LLM ---
python3 "$TRTLLM_DIR/examples/llama/convert_checkpoint.py" \
  --model_dir "$MODEL_DIR" \
  --output_dir "$CKPT_DIR" \
  --dtype float16

# --- Build Engine TensorRT ---
python3 "$TRTLLM_DIR/examples/llama/build.py" \
  --checkpoint_dir "$CKPT_DIR" \
  --output_dir "$ENGINE_DIR" \
  --gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 512 \
  --max_output_len 256

echo "Conversion terminee. Engine: $ENGINE_DIR"
