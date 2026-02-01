#!/bin/bash

# Language Identification Embedding Extraction and Verification
# ===============================================================
#
# Extracts 256D embeddings and computes EER/minDCR for language verification
#
# Usage: bash eval.sh [gpu] [audio_len]
#
# Examples:
#   bash eval.sh                        # Uses all defaults
#   bash eval.sh 0                      # GPU 0 with defaults
#   bash eval.sh 0 160000               # GPU 0, longer audio (10 seconds)

# Get script directory (toolbox root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Python interpreter
PYTHON_ENV="./add-env/bin/python"

if [ -f "$PYTHON_ENV" ]; then
    PYTHON="$PYTHON_ENV"
else
    PYTHON="python"
fi

# ===== CONFIGURATION (Hardcoded paths) =====
DATASET_ROOTS="./TidyVoiceX_Train ./TidyVoiceX_Dev"
CHECKPOINT_DIR="./ckpt_lid/lid_layers17-24_simplehead_bs64_ep15_m0.3_s30.0_h512_w2vLarge"
TRIALS_FILE="verification_trials.txt"
BATCH_SIZE=64
NUM_EPOCHS=15
ARCFACE_MARGIN=0.3
ARCFACE_SCALE=30.0
HIDDEN_DIM=512

# ===== USER OVERRIDES (optional) =====
GPU_ID=${1:-0}
AUDIO_LEN=${2:-64600}

# Trials file path
TRIALS_PATH="${SCRIPT_DIR}/data/trials/${TRIALS_FILE}"
if [ ! -f "$TRIALS_PATH" ]; then
    echo "ERROR: Trials file not found at: $TRIALS_PATH"
    exit 1
fi

echo "======================================================================="
echo "Language ID Embedding Extraction and Verification - TidyLang Baseline"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  - Checkpoint: $CHECKPOINT_DIR"
echo "  - Trials file: $TRIALS_PATH"
echo "  - Dataset roots: $DATASET_ROOTS"
echo "  - GPU: $GPU_ID"
echo "  - Audio length: ${AUDIO_LEN} samples (~$((AUDIO_LEN / 16000)) sec at 16kHz)"
echo "  - Embedding dimension: 256D"
echo "  - Model parameters: bs=$BATCH_SIZE, ep=$NUM_EPOCHS, m=$ARCFACE_MARGIN, s=$ARCFACE_SCALE, h=$HIDDEN_DIM"
echo ""

# Verify checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

$PYTHON "$SCRIPT_DIR/eval.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --trials_file "$TRIALS_PATH" \
    --dataset_roots $DATASET_ROOTS \
    --gpu "$GPU_ID" \
    --batch_size 32 \
    --audio_len "$AUDIO_LEN" \
    --cache_embeddings

echo ""
echo "Evaluation completed!"
