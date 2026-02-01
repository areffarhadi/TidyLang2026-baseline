"""
Language Identification Verification During Training
=====================================================

Extracts embeddings from checkpoint and computes EER using trial pairs.
This script is called after each epoch during training.

Usage:
    python verify_lid_epoch.py \
        --checkpoint /path/to/checkpoint.pt \
        --trials_file language_verification_trials.txt \
        --dataset_roots CV_datasets_wav/multilingual_lists2/TidyVoiceX_Train CV_datasets_wav/multilingual_lists2/TidyVoiceX_Dev \
        --output_dir ./ckpt_lid/output \
        --gpu 0
"""

import warnings
# Suppress FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
import math
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import sys

torch.set_default_dtype(torch.float32)


class Wav2VecLayerExtractor(nn.Module):
    """Extract and aggregate features from Wav2Vec2 layers 17-24."""

    def __init__(self, model_name="facebook/wav2vec2-large"):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Layer indices for layers 17-24
        self.layer_indices = list(range(17, 25))
        
        # Learned weights for layer aggregation
        self.layer_weights = nn.Parameter(torch.ones(len(self.layer_indices)))

    def forward(self, audio_data):
        """Extract and aggregate features from layers 17-24."""
        feat = self.processor(
            audio_data, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(audio_data.device)

        if feat.dim() == 3:
            feat = feat.squeeze(0)

        with torch.no_grad():
            # Get intermediate outputs
            outputs = self.model(
                feat,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract selected layers
            layer_outputs = [outputs.hidden_states[idx] for idx in self.layer_indices]
        
        # Aggregate with learned weights
        weights = F.softmax(self.layer_weights, dim=0)
        aggregated = sum(w * layer_out for w, layer_out in zip(weights, layer_outputs))
        
        return aggregated


class SimpleProjectionHead(nn.Module):
    """Simple Projection Head for Language Identification."""

    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        embedding_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x, normalize=False):
        """Mean pooling and projection."""
        pooled = x.mean(dim=1)  # (B, input_dim)
        embeddings = self.projection(pooled)  # (B, embedding_dim)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class ArcFaceLoss(nn.Module):
    """ArcFace loss for language identification."""
    
    def __init__(self, in_features, out_features, margin=0.3, scale=30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        """Compute ArcFace loss and logits."""
        # Normalize input and weight
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_sim = F.linear(input_norm, weight_norm)  # (B, out_features)
        
        # Compute angles
        cos_angle = torch.clamp(cos_sim, -1, 1)
        angle = torch.acos(cos_angle)
        
        # Add margin to target class angle
        one_hot = F.one_hot(label, num_classes=self.weight.size(0)).float()
        angle_margin = angle + one_hot * self.margin
        
        # Compute logits with scale
        logits = torch.cos(angle_margin) * self.scale
        
        # Compute loss (cross entropy)
        loss = F.cross_entropy(logits, label)
        
        return loss, logits

    def forward_inference(self, input):
        """Forward pass for inference (no label)."""
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(input_norm, weight_norm)
        return logits


class LanguageIDModel(nn.Module):
    """End-to-end Language Identification model."""

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
        self.embedding_dim = embedding_dim

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
        """
        Args:
            waveforms: (B,) audio waveforms
            labels: (B,) language class labels (optional)

        Returns:
            emb_norm: normalized embeddings (B, embedding_dim)
        """
        feat = self.wav2vec_layer24(waveforms)
        emb_norm = self.head(feat, normalize=True)
        
        if labels is not None:
            loss, logits = self.arcface(emb_norm, labels)
            return emb_norm, loss, logits
        else:
            return emb_norm

    def extract_embedding(self, waveforms):
        """Extract normalized embeddings for verification."""
        with torch.no_grad():
            feat = self.wav2vec_layer24(waveforms)
            emb = self.head(feat, normalize=True)
        return emb


def load_audio(file_path, dataset_roots, sr=16000, audio_len=64600):
    """Load and process audio file."""
    full_path = None
    for root in dataset_roots:
        potential_path = os.path.join(root, file_path)
        if os.path.exists(potential_path):
            full_path = potential_path
            break
    
    if full_path is None:
        return None
    
    try:
        waveform, sr_loaded = torchaudio.load(full_path)
        
        if sr_loaded != sr:
            resampler = torchaudio.transforms.Resample(sr_loaded, sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.squeeze()
        
        if len(waveform) < audio_len:
            pad_amount = audio_len - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:audio_len]
        
        return waveform
    
    except Exception as e:
        return None


def compute_eer(y_true, y_score):
    """Compute Equal Error Rate (EER) and threshold."""
    fpr, fnr, threshold = compute_error_rates(y_true, y_score)
    
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = threshold[eer_idx]
    
    return eer, threshold


def compute_error_rates(y_true, y_score):
    """Compute FPR and FNR efficiently using sorted scores (O(n log n))."""
    # Sort scores in descending order
    sorted_indices = np.argsort(-y_score)
    y_sorted = y_true[sorted_indices]
    scores_sorted = y_score[sorted_indices]
    
    # Number of targets and non-targets
    n_targets = np.sum(y_true == 1)
    n_non_targets = np.sum(y_true == 0)
    
    # Initialize
    fpr_list = [0.0]
    fnr_list = [1.0]
    threshold_list = [np.inf]
    
    # Track cumulative false positives and false negatives
    fp = 0
    fn = n_targets
    
    # Iterate through sorted scores
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            fn -= 1
        else:
            fp += 1
        
        # Compute rates
        fpr = fp / n_non_targets if n_non_targets > 0 else 0
        fnr = fn / n_targets if n_targets > 0 else 0
        
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        threshold_list.append(scores_sorted[i])
    
    return np.array(fpr_list), np.array(fnr_list), np.array(threshold_list)


def extract_embeddings_from_trials(model, trials_file, dataset_roots, device, batch_size=32):
    """Extract embeddings for all utterances in trials file."""
    print(f"\n  Loading trials from {trials_file}...")
    
    # Parse trials file and collect unique utterances
    trials = []
    utterances_set = set()
    
    with open(trials_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                # Format: utt1 utt2 label
                # where label is "target" (1) or "nontarget" (0)
                utt1 = parts[0]
                utt2 = parts[1]
                label_str = parts[2].lower()
                
                # Convert label string to binary
                if label_str == "target":
                    label = 1
                elif label_str == "nontarget":
                    label = 0
                else:
                    # Try to parse as int (for compatibility)
                    try:
                        label = int(label_str)
                    except ValueError:
                        continue
                
                trials.append((label, utt1, utt2))
                utterances_set.add(utt1)
                utterances_set.add(utt2)
    
    utterances = sorted(list(utterances_set))
    print(f"  Found {len(utterances)} unique utterances from {len(trials)} trials")
    
    # Load audio and extract embeddings
    embeddings_dict = {}
    failed_utts = []
    
    print(f"  Extracting embeddings...")
    for i in tqdm(range(0, len(utterances), batch_size)):
        batch_utts = utterances[i:i+batch_size]
        
        # Load audio for batch
        waveforms_list = []
        valid_utts = []
        
        for utt in batch_utts:
            waveform = load_audio(utt, dataset_roots)
            if waveform is not None:
                waveforms_list.append(waveform)
                valid_utts.append(utt)
            else:
                failed_utts.append(utt)
        
        if not valid_utts:
            continue
        
        # Stack and move to device
        waveforms_batch = torch.stack(waveforms_list).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            emb = model.extract_embedding(waveforms_batch)
        
        # Store embeddings
        for utt, emb_vec in zip(valid_utts, emb):
            embeddings_dict[utt] = emb_vec.cpu().numpy()
    
    if failed_utts:
        print(f"  Warning: Failed to load {len(failed_utts)} utterances")
    
    # Compute trial scores
    print(f"  Computing trial scores...")
    labels = []
    scores = []
    
    for label, utt1, utt2 in trials:
        if utt1 in embeddings_dict and utt2 in embeddings_dict:
            emb1 = embeddings_dict[utt1]
            emb2 = embeddings_dict[utt2]
            
            # Cosine similarity
            score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            
            labels.append(label)
            scores.append(score)
    
    return np.array(labels), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Language ID Epoch Verification")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--trials_file",
        type=str,
        required=True,
        help="Path to trials file",
    )
    parser.add_argument(
        "--dataset_roots",
        nargs="+",
        required=True,
        help="Dataset root directories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--ssl_model",
        type=str,
        default="facebook/wav2vec2-large",
        help="SSL model name",
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint to get model parameters
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract model parameters from checkpoint
    model_state = checkpoint.get("model_state_dict", checkpoint)
    
    # Count languages from arcface weight shape
    # The arcface weight shape is (num_languages, embedding_dim)
    if "arcface.weight" in model_state:
        num_languages = model_state["arcface.weight"].shape[0]
        embedding_dim = model_state["arcface.weight"].shape[1]
    else:
        print("Error: Cannot find arcface weights in checkpoint")
        sys.exit(1)
    
    print(f"  Model: {num_languages} languages, {embedding_dim}D embeddings")
    
    # Build model
    model = LanguageIDModel(
        num_languages=num_languages,
        ssl_model=args.ssl_model,
        embedding_dim=embedding_dim,
        hidden_dim=512,
        arcface_margin=0.3,
        arcface_scale=30.0,
        device=str(device),
    )
    model.to(device)
    model.eval()
    
    # Load checkpoint
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Extract embeddings and compute EER
    labels, scores = extract_embeddings_from_trials(
        model,
        args.trials_file,
        args.dataset_roots,
        device,
        batch_size=args.batch_size,
    )
    
    # Compute EER
    eer, eer_threshold = compute_eer(labels, scores)
    
    print(f"\n  Equal Error Rate (EER): {100 * eer:.4f}%")
    print(f"  EER Threshold: {eer_threshold:.6f}")
    
    # Log results
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, "verification_eer.log")
    with open(log_file, "a") as f:
        f.write(f"{eer:.6f}\n")
    
    # Also save detailed results
    results = {
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "num_trials": len(labels),
        "num_targets": int(np.sum(labels == 1)),
        "num_non_targets": int(np.sum(labels == 0)),
    }
    
    results_file = os.path.join(args.output_dir, "verification_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {log_file}")
    print(f"  Detailed results: {results_file}")


if __name__ == "__main__":
    main()
