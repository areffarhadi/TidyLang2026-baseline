#!/bin/bash

# Language Identification (LID) Training Script
#
# Training: New speakers in all languages
# Validation: New speakers (flag 2) in all languages
# Validation Crosslingual: Known speakers (flag 3) in different languages
#
# Usage:
#   bash train.sh [gpu_id] [batch_size] [num_epochs] [arcface_margin] [arcface_scale] [hidden_dim]
#   Default: GPU 0, batch_size 64, epochs 15, margin 0.3, scale 30.0, hidden_dim 512
#
# Examples:
#   bash train.sh 0
#   bash train.sh 0 32 50 0.3 30.0 512

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

# Parameters
GPU_ID=${1:-0}
BATCH_SIZE=${2:-64}
NUM_EPOCHS=${3:-15}
ARCFACE_MARGIN=${4:-0.3}
ARCFACE_SCALE=${5:-30.0}
HIDDEN_DIM=${6:-512}

# Relative paths (relative to toolbox root)
UNIFIED_MANIFEST="${SCRIPT_DIR}/data/manifests/training_manifest.txt"
TRIALS_FILE="${SCRIPT_DIR}/data/trials/verification_trials.txt"
MODEL="facebook/wav2vec2-large"
OUTPUT_DIR="${SCRIPT_DIR}/ckpt_lid/lid_layers17-24_simplehead_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_m${ARCFACE_MARGIN}_s${ARCFACE_SCALE}_h${HIDDEN_DIM}_w2vLarge"

echo "======================================================================="
echo "Language Identification Training - TidyLang Baseline"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  - GPU: $GPU_ID"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Hidden dimension: $HIDDEN_DIM"
echo "  - ArcFace:"
echo "      - Margin (m): $ARCFACE_MARGIN"
echo "      - Scale (s): $ARCFACE_SCALE"
echo "  - Verification: Language verification trials"
echo ""
echo "Data Paths:"
echo "  - Manifest: $UNIFIED_MANIFEST"
echo "  - Dataset roots: $DATASET_ROOTS"
echo "  - Trials file: $TRIALS_FILE"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Verify manifest file exists
if [ ! -f "$UNIFIED_MANIFEST" ]; then
    echo "ERROR: Manifest file not found: $UNIFIED_MANIFEST"
    exit 1
fi

# Verify trials file exists
if [ ! -f "$TRIALS_FILE" ]; then
    echo "ERROR: Trials file not found: $TRIALS_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Run training
$PYTHON "$SCRIPT_DIR/main_train.py" \
    --unified_manifest "$UNIFIED_MANIFEST" \
    --dataset_roots $DATASET_ROOTS \
    --trials_file "$TRIALS_FILE" \
    --ssl_model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --arcface_margin "$ARCFACE_MARGIN" \
    --arcface_scale "$ARCFACE_SCALE" \
    --hidden_dim "$HIDDEN_DIM" \
    --embedding_dim 256 \
    --out_fold "$OUTPUT_DIR" \
    --gpu "$GPU_ID"

echo ""
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate the trained model:"
echo "  export DATASET_ROOTS=\"$DATASET_ROOTS\""
echo "  bash eval.sh $BATCH_SIZE $NUM_EPOCHS $ARCFACE_MARGIN $ARCFACE_SCALE $HIDDEN_DIM verification_trials.txt $GPU_ID"
