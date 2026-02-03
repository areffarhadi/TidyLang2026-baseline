#!/usr/bin/env python3
"""
Speaker Verification Evaluation using Enrollment Manifest
=========================================================

Evaluates enrollment-based speaker verification by:
1. Computing similarity scores between test utterances and enrollment utterances
2. Averaging scores across all enrollment utterances for each enrollment ID
3. Computing EER and minDCF metrics

Usage:
    python eval_enrollment_100k.py \
        --checkpoint_dir /path/to/checkpoint \
        --trials_file enrollment_trials_100k.txt \
        --manifest_file enrollment_manifest.tsv \
        --dataset_roots /path/to/audio/root1 /path/to/audio/root2 \
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
from pathlib import Path

torch.set_default_dtype(torch.float32)


class Wav2VecLayerExtractor(nn.Module):
    """Extract and aggregate features from Wav2Vec2 layers 17-24."""

    def __init__(self, model_name="facebook/wav2vec2-large"):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self.layer_indices = list(range(17, 25))
        self.layer_weights = nn.Parameter(torch.ones(len(self.layer_indices)))

    def forward(self, waveforms):
        """Extract features from layers 17-24."""
        if isinstance(waveforms, torch.Tensor):
            waveforms_np = waveforms.cpu().numpy()
        else:
            waveforms_np = waveforms
        
        inputs = self.processor(
            waveforms_np, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        layer_outputs = [hidden_states[idx] for idx in self.layer_indices]
        
        weights = F.softmax(self.layer_weights, dim=0)
        aggregated = sum(w * layer_out for w, layer_out in zip(weights, layer_outputs))
        
        return aggregated


class SimpleProjectionHead(nn.Module):
    """Simple Projection Head for embeddings."""

    def __init__(self, input_dim=1024, hidden_dim=512, embedding_dim=256, dropout=0.1):
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
        """Extract embeddings from features."""
        pooled = x.mean(dim=1)
        embeddings = self.projection(pooled)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class SpeakerVerificationModel(nn.Module):
    """Combined Wav2Vec2 + projection head for speaker verification."""

    def __init__(self, ssl_model="facebook/wav2vec2-large", embedding_dim=256, 
                 hidden_dim=512, device="cuda:0"):
        super().__init__()
        self.device = device
        
        self.wav2vec = Wav2VecLayerExtractor(model_name=ssl_model)
        self.wav2vec.to(device)
        
        self.head = SimpleProjectionHead(
            input_dim=1024,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=0.1
        )
        self.head.to(device)

    def extract_embedding(self, waveforms):
        """Extract embeddings from waveforms."""
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        
        with torch.no_grad():
            feat = self.wav2vec(waveforms)
            embeddings = self.head(feat, normalize=True)
        
        return embeddings


def load_trials(trials_file):
    """Load trials from file format: label enrollment_id test_utterance"""
    trials = []
    with open(trials_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                label = int(parts[0])
                enrollment_id = parts[1]
                test_utterance = parts[2]
                trials.append({
                    'label': label,
                    'enrollment_id': enrollment_id,
                    'test_utterance': test_utterance
                })
    return trials


def load_manifest(manifest_file):
    """Load enrollment manifest from TSV file: enrollment_id file1 file2 ... fileN"""
    manifest = {}
    with open(manifest_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                enrollment_id = parts[0]
                files = parts[1:]
                manifest[enrollment_id] = files
    return manifest


def load_audio(file_path, sr=16000, audio_len=64600):
    """Load and process audio file."""
    try:
        waveform, file_sr = torchaudio.load(file_path)
        
        if waveform.ndim > 2:
            waveform = waveform.squeeze()
        
        if waveform.ndim == 2:
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            else:
                waveform = waveform.squeeze(0)
        
        if file_sr != sr:
            resampler = torchaudio.transforms.Resample(file_sr, sr)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze()
        
        if len(waveform) < audio_len:
            pad_amount = audio_len - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:audio_len]
        
        return waveform
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_error_rates(y_true, y_score):
    """Compute FPR and FNR."""
    sorted_indices = np.argsort(-y_score)
    y_sorted = y_true[sorted_indices]
    scores_sorted = y_score[sorted_indices]
    
    n_targets = np.sum(y_true == 1)
    n_non_targets = np.sum(y_true == 0)
    
    fpr_list = [0.0]
    fnr_list = [1.0]
    threshold_list = [np.inf]
    
    fp = 0
    fn = n_targets
    
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            fn -= 1
        else:
            fp += 1
        
        fpr = fp / n_non_targets if n_non_targets > 0 else 0
        fnr = fn / n_targets if n_targets > 0 else 0
        
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        threshold_list.append(scores_sorted[i])
    
    return np.array(fpr_list), np.array(fnr_list), np.array(threshold_list)


def compute_eer(y_true, y_score):
    """Compute Equal Error Rate (EER)."""
    fpr, fnr, thresholds = compute_error_rates(y_true, y_score)
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]
    return eer, threshold


def compute_mindcf(y_true, y_score, target_prior=0.01):
    """Compute minimum Detection Cost Function (minDCF)."""
    fpr, fnr, thresholds = compute_error_rates(y_true, y_score)
    dcf = target_prior * fnr + (1 - target_prior) * fpr
    mindcf_idx = np.nanargmin(dcf)
    mindcf = dcf[mindcf_idx]
    threshold = thresholds[mindcf_idx]
    return mindcf, threshold


def find_audio_file(relative_path, dataset_roots):
    """Find audio file in dataset roots."""
    for root in dataset_roots:
        full_path = os.path.join(root, relative_path)
        if os.path.exists(full_path):
            return full_path
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate speaker verification using enrollment manifest')
    parser.add_argument('--checkpoint_dir', required=True, help='Checkpoint directory')
    parser.add_argument('--trials_file', required=True, help='Trials file (label enrollment_id test_utterance)')
    parser.add_argument('--manifest_file', required=True, help='Enrollment manifest file (enrollment_id file1 file2 ...)')
    parser.add_argument('--dataset_roots', nargs='+', required=True, help='Dataset root directories')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    parser.add_argument('--audio_len', type=int, default=64600, help='Audio length in samples')
    parser.add_argument('--cache_dir', default='./embeddings_cache', help='Cache directory for embeddings')
    
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    os.makedirs(args.cache_dir, exist_ok=True)

    print("="*80)
    print("SPEAKER VERIFICATION EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Trials file: {args.trials_file}")
    print(f"Manifest file: {args.manifest_file}")
    print(f"Dataset roots: {args.dataset_roots}")
    print(f"Cache directory: {args.cache_dir}\n")

    # Load checkpoint metadata
    print("Loading model...")
    args_file = os.path.join(args.checkpoint_dir, "args.json")
    with open(args_file, 'r') as f:
        ckpt_args = json.load(f)
    
    # Create model
    model = SpeakerVerificationModel(
        ssl_model=ckpt_args.get('ssl_model', 'facebook/wav2vec2-large'),
        embedding_dim=ckpt_args.get('embedding_dim', 256),
        hidden_dim=ckpt_args.get('hidden_dim', 512),
        device=device
    )
    
    # Load checkpoint - try best_checkpoint.pt first, then other names
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        ckpt_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
        if ckpt_files:
            checkpoint_path = os.path.join(args.checkpoint_dir, sorted(ckpt_files)[-1])
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # Disable dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    print("✓ Model loaded\n")

    # Load trials and manifest
    print("Loading trials and manifest...")
    trials = load_trials(args.trials_file)
    manifest = load_manifest(args.manifest_file)
    print(f"✓ Loaded {len(trials)} trials")
    print(f"✓ Loaded {len(manifest)} enrollment IDs\n")

    # Extract all unique utterances
    all_utterances = set()
    for trial in trials:
        all_utterances.add(trial['test_utterance'])
    
    for enr_files in manifest.values():
        for f in enr_files:
            all_utterances.add(f)

    print(f"Extracting embeddings for {len(all_utterances)} unique utterances...")
    embeddings = {}
    
    with tqdm(total=len(all_utterances)) as pbar:
        for utterance in all_utterances:
            cache_key = utterance.replace('/', '_')
            cache_file = os.path.join(args.cache_dir, f"{cache_key}.npy")
            
            if os.path.exists(cache_file):
                embeddings[utterance] = np.load(cache_file)
            else:
                audio_path = find_audio_file(utterance, args.dataset_roots)
                if audio_path:
                    waveform = load_audio(audio_path, audio_len=args.audio_len)
                    if waveform is not None:
                        with torch.no_grad():
                            waveform_tensor = waveform.unsqueeze(0).to(device)
                            embedding = model.extract_embedding(waveform_tensor)
                            embedding_np = embedding.cpu().numpy()
                        embeddings[utterance] = embedding_np
                        np.save(cache_file, embedding_np)
                    else:
                        embeddings[utterance] = np.zeros((1, 256))
                else:
                    print(f"\nWARNING: Audio file not found: {utterance}")
                    embeddings[utterance] = np.zeros((1, 256))
            
            pbar.update(1)

    print(f"✓ Extracted embeddings for {len(embeddings)} utterances\n")

    # Compute trial scores
    print("Computing trial scores...")
    scores = []
    labels = []
    
    with tqdm(total=len(trials)) as pbar:
        for trial in trials:
            enrollment_id = trial['enrollment_id']
            test_utterance = trial['test_utterance']
            label = trial['label']
            
            if enrollment_id not in manifest:
                print(f"\nWARNING: Enrollment ID not found in manifest: {enrollment_id}")
                scores.append(0.0)
                labels.append(label)
                pbar.update(1)
                continue
            
            test_embedding = embeddings.get(test_utterance, np.zeros((1, 256)))
            
            enrollment_utterances = manifest[enrollment_id]
            enrollment_embeddings = []
            
            for enr_utterance in enrollment_utterances:
                enr_embedding = embeddings.get(enr_utterance, np.zeros((1, 256)))
                enrollment_embeddings.append(enr_embedding)
            
            # Compute similarity scores with all enrollment utterances
            similarity_scores = []
            for enr_embedding in enrollment_embeddings:
                test_emb_norm = test_embedding.reshape(-1) / (np.linalg.norm(test_embedding.reshape(-1)) + 1e-8)
                enr_emb_norm = enr_embedding.reshape(-1) / (np.linalg.norm(enr_embedding.reshape(-1)) + 1e-8)
                similarity = np.dot(test_emb_norm, enr_emb_norm)
                similarity_scores.append(similarity)
            
            # Average the scores
            avg_score = np.mean(similarity_scores) if similarity_scores else 0.0
            
            scores.append(avg_score)
            labels.append(label)
            pbar.update(1)

    print(f"✓ Computed scores for {len(scores)} trials\n")

    # Compute metrics
    print("="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    target_scores = scores[labels == 1]
    nontarget_scores = scores[labels == 0]
    
    print(f"Target scores (label=1):")
    print(f"  Mean: {np.mean(target_scores):.6f}")
    print(f"  Std:  {np.std(target_scores):.6f}")
    print(f"  Min:  {np.min(target_scores):.6f}")
    print(f"  Max:  {np.max(target_scores):.6f}\n")
    
    print(f"Nontarget scores (label=0):")
    print(f"  Mean: {np.mean(nontarget_scores):.6f}")
    print(f"  Std:  {np.std(nontarget_scores):.6f}")
    print(f"  Min:  {np.min(nontarget_scores):.6f}")
    print(f"  Max:  {np.max(nontarget_scores):.6f}\n")
    
    # EER
    eer, eer_threshold = compute_eer(labels, scores)
    print(f"Equal Error Rate (EER):")
    print(f"  EER: {eer*100:.4f}%")
    print(f"  Threshold: {eer_threshold:.6f}\n")
    
    # minDCF
    min_dcf, min_dcf_threshold = compute_mindcf(labels, scores)
    print(f"Minimum Detection Cost Function (minDCF):")
    print(f"  minDCF: {min_dcf:.6f}")
    print(f"  Threshold: {min_dcf_threshold:.6f}\n")
    
    print("="*80 + "\n")
    
    # Save results
    results = {
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'min_dcf': float(min_dcf),
        'min_dcf_threshold': float(min_dcf_threshold),
        'target_mean': float(np.mean(target_scores)),
        'target_std': float(np.std(target_scores)),
        'nontarget_mean': float(np.mean(nontarget_scores)),
        'nontarget_std': float(np.std(nontarget_scores)),
        'num_trials': len(trials),
        'num_target': int(np.sum(labels == 1)),
        'num_nontarget': int(np.sum(labels == 0))
    }
    
    results_file = os.path.join(os.path.dirname(args.trials_file), 'eval_results_enrollment.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")


if __name__ == '__main__':
    main()
