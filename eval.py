"""
Language Identification — Evaluation (Identification Only)
==========================================================

Evaluates language identification (classification) on a manifest.
No verification: uses the full model (Wav2Vec2 + head + ArcFace classifier)
to predict language per utterance and reports micro/macro accuracy.

Use eval_enrollment.sh / eval_enrollment.py for verification (unseen languages).

Usage:
    python eval.py \
        --checkpoint_dir ./ckpt_lid/... \
        --manifest_file data/manifests/training_manifest.txt \
        --dataset_roots /path/to/TidyVoiceX_Train /path/to/TidyVoiceX_Dev \
        --split_flag 2 \
        --gpu 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix

torch.set_default_dtype(torch.float32)

# Import ArcFace for the classifier head
from losses import ArcFaceLoss


class Wav2VecLayerExtractor(nn.Module):
    """Extract and aggregate features from Wav2Vec2 layers 17-24 (matches main_train)."""

    def __init__(self, model_name="facebook/wav2vec2-large"):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_indices = list(range(17, 25))
        self.layer_weights = nn.Parameter(torch.ones(len(self.layer_indices)))

    def forward(self, audio_data):
        feat = self.processor(
            audio_data, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(audio_data.device)
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        with torch.no_grad():
            outputs = self.model(feat, output_hidden_states=True, return_dict=True)
        layer_outputs = [outputs.hidden_states[idx] for idx in self.layer_indices]
        weights = F.softmax(self.layer_weights, dim=0)
        aggregated = sum(w * layer_out for w, layer_out in zip(weights, layer_outputs))
        return aggregated


class SimpleProjectionHead(nn.Module):
    """Simple Projection Head (matches main_train)."""

    def __init__(self, input_dim=1024, hidden_dim=512, embedding_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x, normalize=False):
        pooled = x.mean(dim=1)
        embeddings = self.projection(pooled)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class LanguageIDModel(nn.Module):
    """End-to-end Language Identification model (matches main_train)."""

    def __init__(
        self,
        num_languages,
        ssl_model="facebook/wav2vec2-large",
        embedding_dim=256,
        hidden_dim=512,
        arcface_margin=0.3,
        arcface_scale=30.0,
        device="cuda",
    ):
        super().__init__()
        self.num_languages = num_languages
        self.device = device
        self.wav2vec_layer24 = Wav2VecLayerExtractor(model_name=ssl_model)
        self.wav2vec_layer24 = self.wav2vec_layer24.to(device)
        self.head = SimpleProjectionHead(
            input_dim=1024,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=0.1,
        )
        self.head = self.head.to(device)
        self.arcface = ArcFaceLoss(
            in_features=embedding_dim,
            out_features=num_languages,
            margin=arcface_margin,
            scale=arcface_scale,
        )
        self.arcface = self.arcface.to(device)

    def forward(self, waveforms, labels=None):
        feat = self.wav2vec_layer24(waveforms)
        emb_norm = self.head(feat, normalize=True)
        if labels is not None:
            loss, logits = self.arcface(emb_norm, labels)
            return emb_norm, logits, loss
        logits = self.arcface.forward_inference(emb_norm)
        return emb_norm, logits


def load_manifest(manifest_file, dataset_roots, split_flag=None, language_to_idx=None):
    """
    Load manifest: tab-separated flag, file_path, language.
    Returns list of (file_path, language_idx), language_to_idx, idx_to_language.
    """
    samples = []
    language_set = set()
    with open(manifest_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            flag, file_path, language = parts
            if split_flag is not None and flag != str(split_flag):
                continue
            language_set.add(language)
            samples.append((file_path, language))
    if language_to_idx is None:
        language_to_idx = {lang: idx for idx, lang in enumerate(sorted(language_set))}
    idx_to_language = {idx: lang for lang, idx in language_to_idx.items()}
    # Filter samples to languages in language_to_idx and convert to indices
    out = []
    for file_path, language in samples:
        if language not in language_to_idx:
            continue
        out.append((file_path, language_to_idx[language]))
    return out, language_to_idx, idx_to_language


def load_audio(file_path, dataset_roots, sr=16000, audio_len=64600):
    """Load and process audio file from dataset roots."""
    full_path = None
    for root in dataset_roots:
        potential = os.path.join(root, file_path)
        if os.path.exists(potential):
            full_path = potential
            break
    if full_path is None:
        return None
    try:
        waveform, file_sr = torchaudio.load(full_path)
        if waveform.ndim > 2:
            waveform = waveform.squeeze()
        if waveform.ndim == 2:
            waveform = torch.mean(waveform, dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        if file_sr != sr:
            resampler = torchaudio.transforms.Resample(file_sr, sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()
        if len(waveform) < audio_len:
            waveform = F.pad(waveform, (0, audio_len - len(waveform)))
        else:
            waveform = waveform[:audio_len]
        return waveform
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Language Identification — Evaluation (identification only)"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument(
        "--manifest_file",
        type=str,
        required=True,
        help="Manifest file (tab-separated: flag, file_path, language)",
    )
    parser.add_argument(
        "--dataset_roots",
        type=str,
        nargs="+",
        required=True,
        help="Dataset root directories",
    )
    parser.add_argument(
        "--split_flag",
        type=str,
        default="2",
        help="Use only lines with this flag (default: 2 = validation)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--audio_len",
        type=int,
        default=64600,
        help="Audio length in samples (~4 s at 16 kHz)",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    checkpoint_dir = args.checkpoint_dir
    # Load language mapping from checkpoint (idx -> language code)
    lang_map_path = os.path.join(checkpoint_dir, "language_mapping.json")
    if not os.path.exists(lang_map_path):
        raise FileNotFoundError(
            f"Language mapping not found: {lang_map_path}. Train with main_train.py first."
        )
    with open(lang_map_path, "r") as f:
        idx_to_language = json.load(f)
    # JSON keys are strings; convert to int for indexing
    idx_to_language = {int(k): v for k, v in idx_to_language.items()}
    language_to_idx = {v: k for k, v in idx_to_language.items()}
    num_languages = len(idx_to_language)

    # Load manifest (only samples whose language is in the model)
    samples, _, _ = load_manifest(
        args.manifest_file,
        args.dataset_roots,
        split_flag=args.split_flag,
        language_to_idx=language_to_idx,
    )
    if not samples:
        raise ValueError(
            f"No samples found in manifest for split_flag={args.split_flag} "
            f"with languages in checkpoint."
        )
    print(f"Loaded {len(samples)} samples from manifest (split_flag={args.split_flag})\n")

    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if ckpt_files:
            checkpoint_path = os.path.join(checkpoint_dir, sorted(ckpt_files)[-1])
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    num_languages_from_ckpt = state_dict["arcface.weight"].shape[0]
    if num_languages_from_ckpt != num_languages:
        num_languages = num_languages_from_ckpt
        idx_to_language = {i: str(i) for i in range(num_languages)}
        if os.path.exists(lang_map_path):
            with open(lang_map_path, "r") as f:
                idx_to_language = {int(k): v for k, v in json.load(f).items()}

    args_file = os.path.join(checkpoint_dir, "args.json")
    ckpt_args = {}
    if os.path.exists(args_file):
        with open(args_file, "r") as f:
            ckpt_args = json.load(f)
    model = LanguageIDModel(
        num_languages=num_languages,
        ssl_model=ckpt_args.get("ssl_model", "facebook/wav2vec2-large"),
        embedding_dim=ckpt_args.get("embedding_dim", 256),
        hidden_dim=ckpt_args.get("hidden_dim", 512),
        arcface_margin=ckpt_args.get("arcface_margin", 0.3),
        arcface_scale=ckpt_args.get("arcface_scale", 30.0),
        device=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    print(f"Model loaded from {checkpoint_path}\n")

    # Evaluate in batches
    all_preds = []
    all_labels = []
    batch_size = args.batch_size
    num_skipped = 0
    for start in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch_samples = samples[start : start + batch_size]
        waveforms = []
        labels_b = []
        for file_path, lang_idx in batch_samples:
            wav = load_audio(
                file_path, args.dataset_roots, audio_len=args.audio_len
            )
            if wav is None:
                num_skipped += 1
                continue
            waveforms.append(wav)
            labels_b.append(lang_idx)
        if not waveforms:
            continue
        waveforms = torch.stack(waveforms).to(device)
        labels_b = torch.tensor(labels_b, dtype=torch.long, device=device)
        with torch.no_grad():
            _, logits = model(waveforms)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_b.cpu().numpy())
    if num_skipped:
        print(f"Skipped {num_skipped} samples (missing/invalid audio)\n")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)
    correct = (all_preds == all_labels).sum()
    micro_acc = 100.0 * correct / n if n else 0.0
    # Macro: per-class accuracy averaged
    cm = confusion_matrix(
        all_labels, all_preds, labels=list(range(num_languages))
    )
    per_class_acc = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
    macro_acc = 100.0 * per_class_acc.mean()

    print("=" * 60)
    print("RESULTS (Identification Only)")
    print("=" * 60)
    print(f"  Micro Accuracy: {micro_acc:.2f}%")
    print(f"  Macro Accuracy: {macro_acc:.2f}%")
    print(f"  Total samples: {n}")
    print("=" * 60)

    results = {
        "micro_accuracy": float(micro_acc),
        "macro_accuracy": float(macro_acc),
        "num_samples": int(n),
        "num_languages": num_languages,
    }
    results_path = os.path.join(checkpoint_dir, "eval_identification_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
