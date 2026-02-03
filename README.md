# TidyLang-Baseline: Language Identification Training & Evaluation Toolkit

**Official Baseline for the TidyLang 2026 Challenge - Odyssey 2026**

A complete toolbox for training and evaluating a Language Identification (LID) model using Wav2Vec2 + Simple Projection Head + ArcFace Loss. 

## Challenge Information

- **Challenge Website**: https://tidylang2026.github.io/
- **Registration Form**: https://tidylang2026.github.io/10_registration/

For challenge details, rules, and submission guidelines, please visit the official challenge website.

## Directory Structure

```
TidyLang-baseline/
├── README.md
├── requirements.txt
├── losses.py
├── main_train.py
├── train.sh
├── eval.py
├── eval.sh
├── eval_enrollment.py
├── eval_enrollment.sh
├── data/
│   ├── manifests/
│   │   └── training_manifest.txt    # flag, path, language (tab-separated)
│   └── trials/                      # For verification (eval_enrollment only)
│       ├── enrollment_manifest.tsv  # utterances per enrollment ID (20–65 s total per ID)
│       └── trials_Dev.txt           # trial pairs: label enrollment_id test_utterance
└── ckpt_lid/                        # Created by train.sh
    └── lid_layers17-24_simplehead_bs64_ep15_.../
        ├── best_checkpoint.pt
        ├── language_mapping.json
        └── args.json
```

## Quick Start

### Setup



```bash
# Create virtual environment
python -m venv tidylang_env
source tidylang_env/bin/activate

# Install dependencies
cd TidyLang-baseline
pip install -r requirements.txt
```

**Note**: Dataset roots and checkpoint paths are set inside `train.sh`, `eval.sh`, and `eval_enrollment.sh`. Edit those variables to match your machine (no environment variables required).

### Train the Model

```bash
# Train with all default parameters (baseline settings)
cd TidyLang-baseline
bash train.sh

# Optional: Override parameters if needed
bash train.sh 0 32 20 0.3 30.0 512
#            ↓  ↓   ↓   ↓    ↓    ↓
#            |  |   |   |    |    └─ hidden_dim
#            |  |   |   |    └────── arcface_scale
#            |  |   |   └─────────── arcface_margin
#            |  |   └──────────────── num_epochs
#            |  └───────────────────── batch_size
#            └────────────────────────── gpu_id
```

### Monitor Training

Training logs appear in real-time. After each epoch, you'll see validation metrics (identification and language-recognition EER only; no trials file):
```
Epoch 1/15 - Loss: 1.453 | Val Acc: 76.7% | Val CL Acc: 40.3% | Lang EER: 23.68%
```

### Evaluate the Model

- **Identification (35 languages)** — `eval.sh`: predicts language per utterance from the manifest (flag=2) and reports micro/macro accuracy.
- **Verification (5 unseen languages)** — `eval_enrollment.sh`: enrollment-based; computes EER on trial pairs (see [Verification (5 unseen languages)](#verification-5-unseen-languages)).

**Identification:**
```bash
cd TidyLang-baseline
bash eval.sh [gpu] [audio_len] [manifest_file]
# Examples:
#   bash eval.sh              # defaults: GPU 0, 64600 samples, data/manifests/training_manifest.txt
#   bash eval.sh 0 160000     # GPU 0, ~10 s audio
```

**Verification (unseen languages):** Edit paths in `eval_enrollment.sh`, then run `bash eval_enrollment.sh [gpu] [batch_size]`.

## Baseline Results

The TidyLang-Baseline achieves the following performance with default hyperparameters:

| Task | Metric | Value |
|------|--------|------|
| **Identification (35 languages)** | Micro Accuracy | **75.76%** |
| **Identification (35 languages)** | Macro Accuracy | **40.25%** |
| **Verification (5 unseen languages)** | EER | **34.7%** |

- **Identification** is evaluated with `eval.sh` on the validation split (flag=2) of the training manifest.
- **Verification (5 unseen languages)** is evaluated with `eval_enrollment.sh` using enrollment-based trials (embedding per utterance, similarity with each enroll utterance per ID, average score per pair, then EER).

## Training Details

### Input Data Format

**Manifest file** (`data/manifests/training_manifest.txt`):
```
flag    file_path                       language
1       id010001/en/en_30308892.wav    en
1       id010001/en/en_30308891.wav    en
2       id010002/de/de_40923086.wav    de
3       id010003/fr/fr_50123456.wav    fr
...
```

**Data Flags in manifest file:**

| Flag | Dataset | #speaker | Purpose | #lang |
|------|---------|-------|---------|------|
| **1** | Training | 4358 | Train the model to classify languages  | 35 |
| **2** | Validation | 100 | Evaluate classification accuracy and language recognition on new speakers (unseen during training) | 35 |
| **3** | Validation (Cross-lingual) | 100 | Test generalization: evaluate on training speakers but speaking in a different language | 20 |

The model trains on flag=1 data only. After each epoch, it validates on flag=2 (classification accuracy, language-recognition EER) and flag=3 (cross-lingual accuracy). No trials file is used during training.

### Verification (5 unseen languages)

Verification is evaluated separately with **eval_enrollment.sh** (not during training). It uses two files in `data/trials/`:

**1. Enrollment manifest** (`data/trials/enrollment_manifest.tsv`)

- One line per **enrollment ID**.
- Each line lists all **utterance paths** that belong to that enrollment ID (tab-separated): `enrollment_id\tfile1\tfile2\t...`
- The **total duration** of the utterances per enrollment ID is between **20 and 65 seconds** (sum of all files for that ID).
- This file defines “who” each enrollment ID is (the set of enrollment utterances used to compute the enrollment side of the score).

**2. Trial pair file** (`data/trials/trials_Dev.txt`)

- One line per **trial**: `label enrollment_id test_utterance` (space- or tab-separated).
- **label**: `1` = target (test utterance matches the enrollment ID), `0` = non-target (does not match).
- **enrollment_id**: must appear in the enrollment manifest.
- **test_utterance**: path to the test utterance to score against that enrollment ID.

**Scoring pipeline:** Extract an embedding for each utterance → for each trial, compute cosine similarity between the test utterance embedding and **each** enroll utterance of that enrollment ID → **average** those similarities → one score per trial → EER. The 5 unseen languages correspond to the enrollment evaluation setup (e.g. Abkhazian, Hausa, Upper Sorbian, Macedonian, Yoruba).

### Model Architecture

```
Audio Input (16kHz, ~4 seconds)
        ↓
Wav2Vec2-Large
        ↓
Extract Layers 17-24 (1024D each)
        ↓
Aggregate with learned weights → 1024D
        ↓
Simple Projection Head:
  ├─ Linear(1024 → hidden_dim)
  ├─ LayerNorm
  ├─ GELU
  ├─ Dropout(0.1)
  ├─ Linear(hidden_dim → 256)
  ├─ LayerNorm
  └─ L2 Normalization
        ↓
Embeddings (256D)
        ↓
ArcFace Classifier
  ├─ num_classes = number of languages
  ├─ margin = 0.3 (angular margin)
  └─ scale = 30.0 (feature scale)
```

### Training Procedure

1. **Data Loading**:
   - Load flag=1 samples for training (new speakers, all languages)
   - Batch size configurable (default 64)
   - 4 processes for parallel loading

2. **Per Epoch (3 validations)**:
   - **Classification Accuracy (flag=2)**: Macro/Micro accuracy on new speakers
   - **Cross-lingual Accuracy (flag=3)**: Accuracy on known speakers in different languages
   - **Language Recognition EER**: Using flag=2 data, create same-language vs different-language pairs, compute EER



## Understanding Results



### Interpreting Metrics

- **Classification Accuracy**: Higher is better 
- **Language Recognition EER**: Lower is better 



## Evaluation

### Running Evaluation

**Identification (manifest, accuracy):**
```bash
# Edit DATASET_ROOTS and CHECKPOINT_DIR in eval.sh if needed, then:
bash eval.sh [gpu] [audio_len] [manifest_file]
# Example: bash eval.sh 0 64600
```

**Verification (5 unseen languages, enrollment-based EER):** Edit `eval_enrollment.sh` (checkpoint, trials, manifest, dataset roots), then run `bash eval_enrollment.sh [gpu] [batch_size]`.


## Citation

If you use this toolbox in your research, please cite:

```bibtex
@misc{farhadi2026tidy,
      title={TidyVoice: A Curated Multilingual Dataset for Speaker Verification Derived from Common Voice}, 
      author={Aref Farhadipour and Jan Marquenie and Srikanth Madikeri and Eleanor Chodroff},
      year={2026},
      eprint={2601.16358},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2601.16358}, 
}
```



**For help and technical support, please email**: aref.farhadipour@uzh.ch
