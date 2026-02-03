#!/bin/bash

# Language Identification — Evaluation (Identification Only)
# ===========================================================
#
# Evaluates language identification (classification) on a manifest.
# No verification: predicts language per utterance and reports micro/macro accuracy.
# For verification (unseen languages, enrollment-based), use eval_enrollment.sh.
#
# Usage: bash eval.sh [gpu] [audio_len] [manifest_file]
#
# Manifest format: tab-separated flag, file_path, language (same as training).
# By default uses split_flag=2 (validation).
#
# Examples:
#   bash eval.sh                        # Uses all defaults
#   bash eval.sh 0                      # GPU 0 with defaults
#   bash eval.sh 0 160000               # GPU 0, longer audio (~10 s)
#   bash eval.sh 0 64600 data/manifests/my_manifest.txt  # Custom manifest

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PYTHON_ENV="./add-env/bin/python"
if [ -f "$PYTHON_ENV" ]; then
    PYTHON="$PYTHON_ENV"
else
    PYTHON="python"
fi

# ===== CONFIGURATION =====
DATASET_ROOTS="/local/scratch/arfarh/CV_speaker/cv_dataset/CV_datasets_wav/multilingual_lists2/TidyVoiceX_Train /local/scratch/arfarh/CV_speaker/cv_dataset/CV_datasets_wav/multilingual_lists2/TidyVoiceX_Dev"

CHECKPOINT_DIR="/local/scratch/arfarh/CV_speaker/cv_dataset/ckpt_lid/lid_layers17-24_simplehead_bs64_ep15_m0.3_s30.0_h512_w2vLarge"
DEFAULT_MANIFEST="data/manifests/training_manifest.txt"

# ===== USER OVERRIDES =====
GPU_ID=${1:-0}
AUDIO_LEN=${2:-64600}
if [ -n "$3" ]; then
    MANIFEST_PATH="$3"
    [ "${MANIFEST_PATH#/}" = "$MANIFEST_PATH" ] && MANIFEST_PATH="${SCRIPT_DIR}/${MANIFEST_PATH}"
else
    MANIFEST_PATH="${SCRIPT_DIR}/${DEFAULT_MANIFEST}"
fi

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest file not found at: $MANIFEST_PATH"
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "======================================================================="
echo "Language ID — Evaluation (Identification Only)"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  - Checkpoint: $CHECKPOINT_DIR"
echo "  - Manifest:   $MANIFEST_PATH"
echo "  - Dataset roots: $DATASET_ROOTS"
echo "  - GPU: $GPU_ID"
echo "  - Audio length: ${AUDIO_LEN} samples (~$((AUDIO_LEN / 16000)) s at 16kHz)"
echo ""

$PYTHON "$SCRIPT_DIR/eval.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --manifest_file "$MANIFEST_PATH" \
    --dataset_roots $DATASET_ROOTS \
    --split_flag 2 \
    --gpu "$GPU_ID" \
    --batch_size 32 \
    --audio_len "$AUDIO_LEN"

echo ""
echo "Evaluation completed!"
