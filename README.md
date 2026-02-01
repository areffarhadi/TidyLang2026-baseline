# TidyLang-Baseline: Language Identification Training & Evaluation Toolkit

**Official Baseline for the TidyLang 2026 Challenge - Odyssey 2026**

A complete toolbox for training and evaluating a Language Identification (LID) model using Wav2Vec2 (Layers 17-24) + Simple Projection Head + ArcFace Loss. This baseline demonstrates the recommended approach and achieves strong baseline performance on the TidyLang challenge dataset.

## Challenge Information

- **Challenge Website**: https://tidylang2026.github.io/
- **Registration Form**: https://tidylang2026.github.io/10_registration/

For challenge details, rules, and submission guidelines, please visit the official challenge website.



## Directory Structure

```
TidyLang-baseline/
├── README.md                              # Main documentation
├── SETUP.md                               # Setup and installation guide
├── requirements.txt                       # Python dependencies
├── losses.py                              # ArcFace loss implementation
├── main_train.py                          # Training script
├── train.sh                               # Training wrapper (bash)
├── eval.py                                # Evaluation script
├── eval.sh                                # Evaluation wrapper (bash)
├── verify_epoch.py                        # Verification during training
│
├── data/
│   ├── manifests/
│   │   └── training_manifest.txt          # 320K utterances with flags
│   └── trials/
│       └── verification_trials.txt        # 591K trial pairs for verification
│
└── ckpt_lid/                              # Checkpoints (created during training)
    └── lid_layers17-24_simplehead_bs64_ep15_m0.3_s30.0_h512_w2vLarge/
        ├── best_checkpoint.pt
        ├── epoch_0.pt
        ├── val_acc.log
        ├── val_crosslingual_acc.log
        ├── lang_recognition_eer.log
        └── verification_eer.log
```

## Quick Start

### Setup (First Time Only)

See [SETUP.md](SETUP.md) for detailed installation instructions. Quick version:

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
Target (Same Language):
  - Mean cosine similarity: 0.8334
  - Std deviation: 0.1154

Non-target (Different Language):
  - Mean cosine similarity: 0.6482
  - Std deviation: 0.1318

Similarity Gap: 0.1852
Equal Error Rate (EER): 23.68%
Decision Threshold: 0.745
```

### Speaker Verification EER (Language Verification Trials)
```
Equal Error Rate (EER): 36.14%
Decision Threshold: 0.754
```

### Training Summary
```
Train Loss (Final): 1.4530
Train Accuracy: 76.7%
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

**Data Flags Explained:**

| Flag | Dataset | Split | Purpose | Note |
|------|---------|-------|---------|------|
| **1** | Training | New Speakers | Train the model to classify languages on speakers never seen during training | Core training data; highest sample count |
| **2** | Validation | New Speakers | Evaluate classification accuracy and language recognition on new speakers (unseen during training) | Used for model selection and checkpoint saving; reports Micro/Macro accuracy and EER |
| **3** | Validation (Cross-lingual) | Known Speakers, Different Language | Test generalization: evaluate on training speakers but speaking in a different language | Harder task; measures cross-lingual generalization; typically lower accuracy than flag=2 |

The model trains on flag=1 data only. After each epoch, it validates on both flag=2 (domain-matched) and flag=3 (cross-lingual challenge) to monitor generalization.

**Trials file** (`data/trials/verification_trials.txt`):
```
utt1                                utt2                                label
id011063/ab/ab_40923086.wav         id011063/ab/ab_40766731.wav         target
id011063/ab/ab_40923086.wav         id011063/de/de_40804419.wav         nontarget
...
```

Where **label** is "target" (same speaker) or "nontarget" (different speaker).

### Model Architecture

```
Audio Input (16kHz, ~4 seconds)
        ↓
Wav2Vec2-Large [layers 0-24]
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
   - **Speaker Verification EER**: Using language_verification_trials.txt, compute same-speaker vs different-speaker EER

3. **Checkpointing**:
   - Save best_checkpoint.pt (highest validation macro accuracy)
   - Save epoch checkpoints (epoch_0.pt, epoch_1.pt, ...)
   - Log all metrics to separate files

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Batch size for training |
| `num_epochs` | 15 | Number of training epochs |
| `arcface_margin` | 0.3 | Angular margin for ArcFace (larger = harder) |
| `arcface_scale` | 30.0 | Feature scale for ArcFace (larger = sharper boundaries) |
| `hidden_dim` | 512 | Hidden dimension in projection head |
| `embedding_dim` | 256 | Final embedding dimension (fixed) |
| `learning_rate` | 0.0001 | Initial learning rate (AdamW) |
| `weight_decay` | 0.001 | Weight decay for AdamW |
| `warmup_steps` | 1000 | Linear warmup steps |
| `dropout` | 0.1 | Dropout in projection head |

## Understanding Results

### Log Files

After training, check the checkpoint folder for:

1. **val_acc.log**: Classification accuracy on flag=2 (new speakers)
   ```
   Epoch 0: macro_acc=65.23, micro_acc=78.15
   Epoch 1: macro_acc=72.45, micro_acc=81.23
   ...
   ```

2. **val_crosslingual_acc.log**: Accuracy on flag=3 (cross-lingual)
   ```
   Epoch 0: macro_acc=32.11, micro_acc=45.67
   Epoch 1: macro_acc=38.92, micro_acc=52.34
   ...
   ```

3. **lang_recognition_eer.log**: Language recognition EER (flag=2 pairs)
   ```
   Epoch 0: eer=22.34%
   Epoch 1: eer=18.92%
   ...
   ```

4. **verification_eer.log**: Speaker verification EER (language_verification_trials.txt)
   ```
   Epoch 0: eer=37.23%
   Epoch 1: eer=32.15%
   ...
   ```

### Interpreting Metrics

- **Classification Accuracy**: Higher is better (aim for >85% macro accuracy on flag=2)
- **Cross-lingual Accuracy**: Shows generalization to new languages (often lower than classification accuracy)
- **Language Recognition EER**: Lower is better (aim for <15% EER for easy language pairs)
- **Speaker Verification EER**: Lower is better (typically 20-40% depending on data and speaker similarity)

### EER (Equal Error Rate) Explanation

EER occurs where False Positive Rate (FPR) = False Negative Rate (FNR):
- **FPR**: Incorrectly accepting non-matching pairs (false alarms)
- **FNR**: Incorrectly rejecting matching pairs (misses)

Lower EER = Better performance at perfect balance.

## Evaluation

### Running Evaluation

```bash
export DATASET_ROOTS="/path/to/audio/data"
bash eval.sh 64 15 0.3 30.0 512 verification_trials.txt 0 64600
```

### Evaluation Output

The evaluation script will:
1. Search for checkpoints in the default location
2. Load best_checkpoint.pt (highest validation accuracy)
3. Extract 256D embeddings for all utterances in the trials file
4. Compute cosine similarity between all trial pairs
5. Compute EER, minDCR, and other metrics
6. Display results with ROC and DET curves (if matplotlib available)

### Embedding Cache

Evaluation caches embeddings to `embeddings_cache.pkl` for faster re-evaluation.

## Advanced Usage

### Using Different Data

To use your own data:
1. Create a manifest file in format: `flag\tfile_path\tlanguage`
2. Prepare audio files in the same structure
3. Update `data/manifests/training_manifest.txt` or create a new one
4. Run training with the new manifest:
   ```bash
   python main_train.py --unified_manifest /path/to/new_manifest.txt \
                        --dataset_roots /path/to/audio/folder1 /path/to/audio/folder2 \
                        --trials_file /path/to/trials.txt \
                        ...
   ```

### Using Different Audio Length

By default, audio is processed as 64600 samples (~4.04 seconds at 16kHz). To use different lengths:

```bash
# Training: automatically uses 64600 samples
bash train.sh 0

# Evaluation: specify audio length (in samples)
bash eval.sh 64 15 0.3 30.0 512 verification_trials.txt 0 160000  # ~10 seconds
```

### Training on CPU

```bash
CUDA_VISIBLE_DEVICES="" python main_train.py \
    --unified_manifest data/manifests/training_manifest.txt \
    --dataset_roots /path/to/audio/data \
    --trials_file data/trials/verification_trials.txt \
    --gpu -1
```

## Troubleshooting

See [SETUP.md](SETUP.md#troubleshooting) for common setup issues.

### Training is slow

1. **Check GPU utilization**: `nvidia-smi`
2. **Reduce validation frequency**: Modify `main_train.py` to validate every N epochs instead of every epoch
3. **Use mixed precision**: Add `--use_amp` flag (requires PyTorch 1.6+)
4. **Increase batch size**: `bash train.sh 0 128` (if GPU memory allows)

### Low accuracy on cross-lingual data

This is expected - the model is trained on new speakers only, so it may not generalize well to known speakers. To improve:
1. Include flag=3 samples in training (modify manifest to use flag=1 for all data)
2. Increase model capacity (larger hidden_dim)
3. Train for more epochs
4. Use different data augmentation strategies

### Evaluation EER different from training EER

This is normal! Reasons:
- **Training EER**: Uses epoch_XXX.pt checkpoint (current epoch during training)
- **Evaluation EER**: Uses best_checkpoint.pt (best validation accuracy checkpoint)
- Different checkpoints have different performance on the same trials

To match them, either:
1. Evaluate using epoch checkpoints
2. Or re-train the best checkpoint and evaluate

## References

- **Wav2Vec2**: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477) - Self-supervised speech representation learning
- **ArcFace**: [Deng et al., 2019](https://arxiv.org/abs/1801.07698) - Additive angular margin loss for deep face recognition
- **EER**: [Martin et al., 1997](http://www.itl.nist.gov/iad/894.01/tests/sre/1997/speech_rec_eval_plan-97.txt) - NIST speaker recognition evaluation

## License

Please see LICENSE file (if included) or contact authors for licensing information.

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@misc{tidylang-baseline,
  title={TidyLang-Baseline: Language Identification with Wav2Vec2 and ArcFace},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/[your-repo]/tidylang-baseline}}
}
```

## Support

For issues or questions:
1. Check [SETUP.md](SETUP.md#troubleshooting) for common problems
2. Review training logs in `ckpt_lid/` folder
3. Verify data paths and manifest file format
4. Check GPU memory with `nvidia-smi`

**For help and technical support, please email**: aref.farhadipour@uzh.ch
