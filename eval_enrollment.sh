#!/bin/bash

# Language verification Script
# =======================================
#
# Evaluates language verification using enrollment-based trial pairs
# Computes EER and minDCF metrics
#
# Usage: bash eval_enrollment_100k.sh [gpu] [batch_size]
#
# Examples:
#   bash eval_enrollment_100k.sh                    # Uses GPU 0, batch size 32
#   bash eval_enrollment_100k.sh 0                  # GPU 0 with defaults
#   bash eval_enrollment_100k.sh 0 64               # GPU 0, batch size 64

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Python interpreter
PYTHON="/local/scratch/arfarh/wildspoof_challenge/All-Type-ADD/add-env/bin/python"

if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

# ===== CONFIGURATION =====
CHECKPOINT_DIR="/local/scratch/arfarh/CV_speaker/cv_dataset/ckpt_lid/lid_layers17-24_simplehead_bs64_ep15_m0.3_s30.0_h512_w2vLarge"
TRIALS_FILE="${SCRIPT_DIR}/data/trials/trials_Dev.txt"
MANIFEST_FILE="${SCRIPT_DIR}/data/trials/enrollment_manifest.tsv"
DATASET_ROOTS="/local/scratch/arfarh/CV_speaker/cv_dataset/CV_datasets_wav/multilingual_lists2/TidyVoiceX_Train /local/scratch/arfarh/CV_speaker/cv_dataset/CV_datasets_wav/multilingual_lists2/TidyVoiceX_Dev"
CACHE_DIR="${SCRIPT_DIR}/embeddings_cache"

# ===== USER OVERRIDES =====
GPU_ID=${1:-0}
BATCH_SIZE=${2:-32}

# Verify files exist
if [ ! -f "$CHECKPOINT_DIR/best_checkpoint.pt" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_DIR/best_checkpoint.pt"
    exit 1
fi

if [ ! -f "$TRIALS_FILE" ]; then
    echo "ERROR: Trials file not found at: $TRIALS_FILE"
    exit 1
fi

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "ERROR: Manifest file not found at: $MANIFEST_FILE"
    exit 1
fi

echo "======================================================================="
echo "languageVerification Evaluation - Enrollment-based"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  - Checkpoint: $CHECKPOINT_DIR"
echo "  - Trials file: $TRIALS_FILE"
echo "  - Manifest file: $MANIFEST_FILE"
echo "  - Dataset roots: $DATASET_ROOTS"
echo "  - GPU: $GPU_ID"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Cache directory: $CACHE_DIR"
echo ""

# Run evaluation (use full path to Python)
/local/scratch/arfarh/wildspoof_challenge/All-Type-ADD/add-env/bin/python "$SCRIPT_DIR/eval_enrollment.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --trials_file "$TRIALS_FILE" \
    --manifest_file "$MANIFEST_FILE" \
    --dataset_roots $DATASET_ROOTS \
    --gpu "$GPU_ID" \
    --batch_size "$BATCH_SIZE" \
    --cache_dir "$CACHE_DIR"

echo ""
echo "Evaluation completed!"
echo "Results saved to: ${SCRIPT_DIR}/eval_results.json"
