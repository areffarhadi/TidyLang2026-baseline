# TidyLang-Baseline: Language Identification Training & Evaluation Toolkit

**Official Baseline for the TidyLang 2026 Challenge - Odyssey 2026**

A complete toolbox for training and evaluating a Language Identification (LID) model using Wav2Vec2 + Simple Projection Head + ArcFace Loss. 

## Challenge Information

- **Challenge Website**: https://tidylang2026.github.io/
- **Registration Form**: https://tidylang2026.github.io/10_registration/

For challenge details, rules, and submission guidelines, please visit the official challenge website.




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

**Note**: Dataset paths and Python environment are pre-configured in the scripts. No environment variables needed!

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

Training logs appear in real-time. After each epoch, you'll see validation metrics:
```
Epoch 1/15 - Loss: 1.453 | Val Acc: 76.7% | Val CL Acc: 40.3% | Lang EER: 23.68% | Verif EER: 36.14%
```

### Evaluate the Model

```bash
# Evaluate with all default parameters
cd TidyLang-baseline
bash eval.sh

# Optional: Override specific parameters
bash eval.sh 0 160000
#            ↓  ↓
#            |  └─ audio_length (samples at 16kHz, default: 64600)
#            └──── gpu_id (default: 0)
```

## Baseline Results

The TidyLang-Baseline achieves the following performance with default hyperparameters:

### Classification Accuracy (flag=2: New Speakers)
```
Micro Accuracy: 75.8%
Macro Accuracy: 40.3%
```

### Cross-lingual Generalization (flag=3: Known Speakers, Different Language)
```
Micro Accuracy: 58.5%
Macro Accuracy: 57.7%
```

### Language Recognition Performance (flag=2 Data - Same Language vs Different Language)
```

Equal Error Rate (EER): 23.68%
Decision Threshold: 0.745
```

### Speaker Verification EER (Language Verification Trials for 5 languages)
```
Equal Error Rate (EER): 30.7%
Decision Threshold: 0.754
```

## Quick Start

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

The model trains on flag=1 data only. After each epoch, it validates on both flag=2 (domain-matched) and flag=3 (cross-lingual challenge) to monitor generalization.

### Verification Trials

The verification trials file (`data/trials/trials_val_lang.txt`) contains trial pairs for evaluating both **language verification** and **speaker verification** performance. The trials use multilingual speakers to create a comprehensive evaluation across 4 distinct categories.

**Trial Composition:**
- **16 multilingual speakers** (speakers who recorded in multiple languages)
- **5 languages**: Abkhazian (ab), Hausa (ha), Upper Sorbian (hsb), Macedonian (mk), Yoruba (yo)

**Four Types of Trial Pairs:**

The trial pairs are organized into 4 categories based on whether the speaker and language match:

| Category | Speaker | Language | Label | Purpose |
|----------|---------|----------|-------|---------|
| **Type 1** | Same | Same | `target` | Within-speaker, within-language verification |
| **Type 2** | Different | Same | `target` | Cross-speaker, within-language verification |
| **Type 3** | Same | Different | `nontarget` | Within-speaker, cross-lingual discrimination |
| **Type 4** | Different | Different | `nontarget` | Cross-speaker, cross-lingual discrimination |



**Trials file format** (`data/trials/trials_val_lang.txt`):
```
utt1                                utt2                                label
id011063/ab/ab_40923086.wav         id011063/ab/ab_40766731.wav         target
id011063/ab/ab_40923086.wav         id010652/ha/ha_44103062.wav         nontarget
...
```

**Evaluation Insights:**
- **Type 1 + Type 2 (same language)**: Tests language recognition within the same language (should be high similarity)
- **Type 3 (same speaker, different languages)**: Tests if model can distinguish languages from the same speaker (crucial for language ID on multilingual speakers)
- **Type 4 (different speaker, different languages)**: Tests discrimination when both speaker and language differ (easiest case)

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

2. **Per Epoch (4 validations)**:
   - **Classification Accuracy (flag=2)**: Macro/Micro accuracy on new speakers
   - **Cross-lingual Accuracy (flag=3)**: Accuracy on known speakers in different languages
   - **Language Recognition EER**: Using flag=2 data, create same-language vs different-language pairs, compute EER
   - **Speaker Verification EER**: Using trials_val_lang.txt, compute same-speaker vs different-speaker EER



## Understanding Results



### Interpreting Metrics

- **Classification Accuracy**: Higher is better 
- **Language Recognition EER**: Lower is better 



## Evaluation

### Running Evaluation

```bash
export DATASET_ROOTS="/path/to/audio/data"
bash eval.sh 64 15 0.3 30.0 512 trials_val_lang.txt 0 64600
```


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
