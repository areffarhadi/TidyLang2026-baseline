"""
Language Identification Embedding Extraction and Verification
==============================================================

Extracts 256D embeddings from trained model and computes EER/minDCR
for language verification trials.

Usage:
    python eval_lid_simple_head.py \
        --checkpoint_dir ./ckpt_lid/lid_layers17-24_simplehead_bs64_ep15_m0.3_s30.0_h512_w2vLarge \
        --trials_file language_verification_trials.txt \
        --dataset_roots CV_datasets_wav/multilingual_lists2/TidyVoiceX_Train CV_datasets_wav/multilingual_lists2/TidyVoiceX_Dev \
        --gpu 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
import math
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config
import torchaudio
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from collections import defaultdict
import pickle

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
        
        # Learnable weights for aggregation
        self.layer_weights = nn.Parameter(torch.ones(len(self.layer_indices)))

    def forward(self, waveforms):
        """
        Args:
            waveforms: (B, T) audio waveforms at 16kHz

        Returns:
            aggregated: (B, T, 1024) aggregated features from layers 17-24
        """
        # Convert tensor to numpy for processor
        if isinstance(waveforms, torch.Tensor):
            waveforms_np = waveforms.cpu().numpy()
        else:
            waveforms_np = waveforms
        
        # Extract features - processor expects numpy arrays
        inputs = self.processor(
            waveforms_np, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get hidden states
        hidden_states = outputs.hidden_states
        
        # Extract specified layers
        layer_outputs = [hidden_states[idx] for idx in self.layer_indices]
        
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
        """
        Args:
            x: (B, T, input_dim) - sequence of features
            normalize: whether to L2 normalize embeddings

        Returns:
            embeddings: (B, embedding_dim)
        """
        # Mean pooling over time dimension
        pooled = x.mean(dim=1)  # (B, input_dim)

        # Project to embedding
        embeddings = self.projection(pooled)  # (B, embedding_dim)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class LIDModel(nn.Module):
    """Language ID Model for embedding extraction."""

    def __init__(
        self,
        ssl_model="facebook/wav2vec2-large",
        embedding_dim=256,
        hidden_dim=512,
        device="cuda",
    ):
        super().__init__()

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

    def extract_embedding(self, waveforms, return_unnormalized=False):
        """
        Extract normalized embeddings.
        
        Args:
            waveforms: (B, T) or (T,) audio waveforms
            return_unnormalized: If True, also return unnormalized embeddings
            
        Returns:
            embeddings: (B, 256) normalized embeddings
            Or: (normalized, unnormalized) if return_unnormalized=True
        """
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        
        # Extract features
        feat = self.wav2vec_layer24(waveforms)

        # Get normalized embeddings
        embeddings_norm = self.head(feat, normalize=True)
        
        if return_unnormalized:
            embeddings_unnorm = self.head(feat, normalize=False)
            return embeddings_norm, embeddings_unnorm
        else:
            return embeddings_norm


def load_audio(file_path, sr=16000, audio_len=64600):
    """Load and process audio file."""
    try:
        waveform, file_sr = torchaudio.load(file_path)
        
        # Ensure shape is [channels, samples]
        if waveform.ndim > 2:
            waveform = waveform.squeeze()
        
        # Convert to mono if needed
        if waveform.ndim == 2:
            if waveform.shape[0] > 1:
                # Multiple channels - take mean
                waveform = torch.mean(waveform, dim=0)
            else:
                # Single channel - squeeze
                waveform = waveform.squeeze(0)
        
        # Resample if needed
        if file_sr != sr:
            resampler = torchaudio.transforms.Resample(file_sr, sr)
            waveform = resampler(waveform)
        
        # Ensure 1D
        waveform = waveform.squeeze()
        
        # Pad or trim to required length
        if len(waveform) < audio_len:
            pad_amount = audio_len - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:audio_len]
        
        return waveform
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate (EER).
    
    Args:
        y_true: Ground truth labels (1 = target/same, 0 = non-target/different)
        y_score: Similarity scores
        
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
    """
    fpr, fnr, thresholds = compute_error_rates(y_true, y_score)
    
    # Find threshold where FPR == FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]
    
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
    
    # Initialize: at threshold = infinity, all predictions are 0 (accept nothing)
    # FP = 0 (no false positives - no non-targets accepted)
    # FN = n_targets (all targets are false negatives - rejected)
    fpr_list = [0.0]  # FP = 0, so FPR = 0
    fnr_list = [1.0]  # FN = n_targets, so FNR = 1
    threshold_list = [np.inf]
    
    # Track cumulative false positives and false negatives
    fp = 0              # Start with 0 false positives
    fn = n_targets      # Start with all targets as false negatives
    
    # Iterate through sorted scores (from highest to lowest)
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:  # Target
            fn -= 1  # Accept a target, so decrease false negatives
        else:  # Non-target
            fp += 1  # Accept a non-target, so increase false positives
        
        # Compute rates
        fpr = fp / n_non_targets if n_non_targets > 0 else 0
        fnr = fn / n_targets if n_targets > 0 else 0
        
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        threshold_list.append(scores_sorted[i])
    
    return np.array(fpr_list), np.array(fnr_list), np.array(threshold_list)


def compute_mindcr(y_true, y_score, target_prior=0.01):
    """
    Compute minimum Detection Cost Rate (minDCR).
    
    Args:
        y_true: Ground truth labels (1 = target, 0 = non-target)
        y_score: Similarity scores
        target_prior: Prior probability of target (default 0.01)
        
    Returns:
        mindcr: Minimum Detection Cost Rate
        threshold: Threshold at minDCR
    """
    fpr, fnr, thresholds = compute_error_rates(y_true, y_score)
    
    # Cost function: DCR = target_prior * fnr + (1 - target_prior) * fpr
    dcr = target_prior * fnr + (1 - target_prior) * fpr
    
    mindcr_idx = np.nanargmin(dcr)
    mindcr = dcr[mindcr_idx]
    threshold = thresholds[mindcr_idx]
    
    return mindcr, threshold


def main():
    parser = argparse.ArgumentParser(description="Language Identification Embedding Extraction and Evaluation")
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--trials_file",
        type=str,
        required=True,
        help="Path to trials file (format: wav1 wav2 label)",
    )
    parser.add_argument(
        "--dataset_roots",
        type=str,
        nargs="+",
        required=True,
        help="Dataset root directories",
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
        "--audio_len",
        type=int,
        default=64600,
        help="Audio length in samples (default: 64600 = ~4 sec at 16kHz, try 160000 for ~10 sec)",
    )
    parser.add_argument(
        "--cache_embeddings",
        action="store_true",
        help="Cache embeddings to disk",
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load checkpoint metadata
    args_file = os.path.join(args.checkpoint_dir, "args.json")
    with open(args_file, 'r') as f:
        ckpt_args = json.load(f)
    
    print("Loading model...")
    model = LIDModel(
        ssl_model=ckpt_args['ssl_model'],
        embedding_dim=ckpt_args['embedding_dim'],
        hidden_dim=ckpt_args.get('hidden_dim', 512),
        device=device,
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        # Try to find latest checkpoint
        ckpt_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
        if ckpt_files:
            checkpoint_path = os.path.join(args.checkpoint_dir, sorted(ckpt_files)[-1])
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # Disable dropout layers for deterministic inference
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    print("Reading trials file...")
    trials = []
    trial_labels = defaultdict(list)
    
    with open(args.trials_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                wav1, wav2, label = parts[0], parts[1], parts[2]
                trials.append((wav1, wav2, label))
                trial_labels[label].append(len(trials) - 1)
    
    print(f"Loaded {len(trials)} trials")
    print(f"Target trials: {len(trial_labels.get('target', []))}")
    print(f"Non-target trials: {len(trial_labels.get('nontarget', []))}")
    
    # Extract all unique audio files
    unique_audios = set()
    for wav1, wav2, _ in trials:
        unique_audios.add(wav1)
        unique_audios.add(wav2)
    unique_audios = sorted(list(unique_audios))
    
    print(f"\nExtracting embeddings for {len(unique_audios)} unique audio files...")
    
    # Cache embeddings
    embeddings_cache = {}
    
    # Batch processing
    batch_size = args.batch_size
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(unique_audios), batch_size), desc="Extracting (batched)"):
            batch_end = min(batch_start + batch_size, len(unique_audios))
            batch_files = unique_audios[batch_start:batch_end]
            
            # Load batch of audio files
            batch_waveforms = []
            batch_file_indices = []
            
            for file_idx, audio_file in enumerate(batch_files):
                # Find audio in dataset roots
                full_path = None
                for root in args.dataset_roots:
                    potential_path = os.path.join(root, audio_file)
                    if os.path.exists(potential_path):
                        full_path = potential_path
                        break
                
                if full_path is None:
                    embeddings_cache[audio_file] = None
                    continue
                
                # Load audio
                waveform = load_audio(full_path, audio_len=args.audio_len)
                if waveform is None:
                    embeddings_cache[audio_file] = None
                    continue
                
                batch_waveforms.append(waveform)
                batch_file_indices.append((file_idx, audio_file))
            
            if not batch_waveforms:
                continue
            
            # Pad batch to same length
            max_len = max(len(w) for w in batch_waveforms)
            padded_batch = []
            for w in batch_waveforms:
                if len(w) < max_len:
                    w = torch.nn.functional.pad(w, (0, max_len - len(w)))
                padded_batch.append(w)
            
            batch_tensor = torch.stack(padded_batch).to(device)  # (B, T)
            
            # Extract embeddings
            batch_embeddings = model.extract_embedding(batch_tensor)  # (B, 256)
            
            # Cache embeddings
            for (file_idx, audio_file), embedding in zip(batch_file_indices, batch_embeddings):
                embeddings_cache[audio_file] = embedding.cpu().numpy()  # (256,)
    
    # Compute and display embedding statistics
    valid_embeddings = np.array([e for e in embeddings_cache.values() if e is not None])
    
    print(f"\nSuccessfully extracted embeddings for {len(valid_embeddings)} files")
    
    if len(valid_embeddings) > 0:
        print("\nEmbedding Statistics (256D):")
        print(f"  Mean embedding norm: {np.linalg.norm(valid_embeddings, axis=1).mean():.4f}")
        print(f"  Std embedding norm: {np.linalg.norm(valid_embeddings, axis=1).std():.4f}")
        print(f"  Mean dimension value: {valid_embeddings.mean():.6f}")
        print(f"  Std dimension value: {valid_embeddings.std():.6f}")
        print(f"  Min dimension value: {valid_embeddings.min():.6f}")
        print(f"  Max dimension value: {valid_embeddings.max():.6f}")
        print(f"  Variance (per dimension):")
        dim_vars = np.var(valid_embeddings, axis=0)
        print(f"    Mean variance: {dim_vars.mean():.6f}")
        print(f"    Min variance: {dim_vars.min():.6f}")
        print(f"    Max variance: {dim_vars.max():.6f}")
    
    
    # Compute scores for trials - BATCHED
    print("\nComputing trial scores (batched)...")
    scores = []
    labels = []
    
    batch_size_scoring = 10000  # Process trials in batches
    
    for batch_start in tqdm(range(0, len(trials), batch_size_scoring), desc="Scoring (batched)"):
        batch_end = min(batch_start + batch_size_scoring, len(trials))
        batch_trials = trials[batch_start:batch_end]
        
        # Collect embeddings for this batch
        batch_emb1_list = []
        batch_emb2_list = []
        batch_labels = []
        valid_indices = []
        
        for idx, (wav1, wav2, label) in enumerate(batch_trials):
            emb1 = embeddings_cache.get(wav1)
            emb2 = embeddings_cache.get(wav2)
            
            if emb1 is None or emb2 is None:
                continue
            
            batch_emb1_list.append(emb1)
            batch_emb2_list.append(emb2)
            batch_labels.append(1 if label == 'target' else 0)
            valid_indices.append(idx)
        
        if not batch_emb1_list:
            continue
        
        # Stack embeddings
        batch_emb1 = np.array(batch_emb1_list)  # (N, 256)
        batch_emb2 = np.array(batch_emb2_list)  # (N, 256)
        
        # Compute cosine similarities in batch
        # Normalize embeddings
        emb1_norm = np.linalg.norm(batch_emb1, axis=1, keepdims=True)
        emb2_norm = np.linalg.norm(batch_emb2, axis=1, keepdims=True)
        
        batch_emb1_normalized = batch_emb1 / (emb1_norm + 1e-8)
        batch_emb2_normalized = batch_emb2 / (emb2_norm + 1e-8)
        
        # Compute dot products (cosine similarity)
        batch_scores = np.sum(batch_emb1_normalized * batch_emb2_normalized, axis=1)
        
        scores.extend(batch_scores.tolist())
        labels.extend(batch_labels)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    print(f"\nEvaluation on {len(scores)} trials:")
    print(f"  Target (1) trials: {np.sum(labels == 1)}")
    print(f"  Non-target (0) trials: {np.sum(labels == 0)}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean target score: {scores[labels == 1].mean():.4f}")
    print(f"  Mean non-target score: {scores[labels == 0].mean():.4f}")
    print(f"  Target std: {scores[labels == 1].std():.4f}")
    print(f"  Non-target std: {scores[labels == 0].std():.4f}")
    
    # Compute EER
    eer, eer_threshold = compute_eer(labels, scores)
    print(f"\nEqual Error Rate (EER): {100 * eer:.4f}%")
    print(f"  EER Threshold: {eer_threshold:.4f}")
    
    # Compute minDCR
    mindcr, mindcr_threshold = compute_mindcr(labels, scores)
    print(f"\nMinimum Detection Cost Rate (minDCR): {mindcr:.4f}")
    print(f"  minDCR Threshold: {mindcr_threshold:.4f}")
    
    # Save results
    results = {
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'mindcr': float(mindcr),
        'mindcr_threshold': float(mindcr_threshold),
        'num_trials': len(scores),
        'num_target_trials': int(np.sum(labels == 1)),
        'num_nontarget_trials': int(np.sum(labels == 0)),
        'mean_target_score': float(scores[labels == 1].mean()),
        'mean_nontarget_score': float(scores[labels == 0].mean()),
    }
    
    results_file = os.path.join(args.checkpoint_dir, "verification_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Optional: save scores for further analysis
    if args.cache_embeddings:
        scores_file = os.path.join(args.checkpoint_dir, "trial_scores.npz")
        np.savez(scores_file, scores=scores, labels=labels, trials=unique_audios)
        print(f"Scores saved to {scores_file}")


if __name__ == "__main__":
    main()
