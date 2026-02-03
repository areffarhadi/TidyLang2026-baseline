"""
Language Identification (LID) with Layers 17-24 + Multi-Head Attention + ArcFace
=================================================================================

Uses unified manifest with three dataset splits (flags):
- Flag 1: Training (new speakers)
- Flag 2: Validation (new speakers)
- Flag 3: Validation Crosslingual (known speakers in different languages)

Computes both micro and macro-averaged accuracy for each validation set after each epoch.
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
import sys
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
from losses import ArcFaceLoss
from sklearn.metrics import accuracy_score, confusion_matrix, confusion_matrix

torch.set_default_dtype(torch.float32)


class UnifiedLanguageDataset(Dataset):
    """Dataset for language identification from unified manifest file with split flags."""
    
    def __init__(self, manifest_file, dataset_roots, split_flag=None, sr=16000, audio_len=64600, language_to_idx=None):
        """
        Args:
            manifest_file: Path to unified manifest (tab-separated: flag, path, language)
            dataset_roots: List of dataset root directories
            split_flag: Filter by flag (1=train, 2=val, 3=val_crosslingual), None=all
            sr: Sampling rate
            audio_len: Audio length in samples
            language_to_idx: Optional pre-built language mapping (for validation/test sets)
        """
        self.samples = []
        self.language_to_idx = language_to_idx if language_to_idx is not None else {}
        self.idx_to_language = {}
        self.sr = sr
        self.audio_len = audio_len
        self.dataset_roots = dataset_roots
        self.split_flag = split_flag
        
        self._load_manifest(manifest_file)
    
    def _load_manifest(self, manifest_file):
        """Load and parse manifest file."""
        language_set = set()
        
        with open(manifest_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 3:
                    continue
                
                flag, file_path, language = parts
                
                # Filter by split flag if specified
                if self.split_flag is not None and flag != str(self.split_flag):
                    continue
                
                language_set.add(language)
                self.samples.append((file_path, language))
        
        # Create language mapping only if not provided
        if not self.language_to_idx:
            for idx, language in enumerate(sorted(language_set)):
                self.language_to_idx[language] = idx
        
        # Build reverse mapping
        for language, idx in self.language_to_idx.items():
            self.idx_to_language[idx] = language
        
        print(f"  Loaded {len(self.samples)} samples")
        print(f"  Languages in this set: {sorted(language_set)}")
    
    def _load_audio(self, file_path):
        """Load and process audio file."""
        full_path = None
        for root in self.dataset_roots:
            potential_path = os.path.join(root, file_path)
            if os.path.exists(potential_path):
                full_path = potential_path
                break
        
        if full_path is None:
            return torch.zeros(self.audio_len)
        
        try:
            waveform, sr = torchaudio.load(full_path)
            
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.squeeze()
            
            if len(waveform) < self.audio_len:
                pad_amount = self.audio_len - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            else:
                waveform = waveform[:self.audio_len]
            
            return waveform
        
        except Exception as e:
            return torch.zeros(self.audio_len)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, language = self.samples[idx]
        waveform = self._load_audio(file_path)
        language_idx = self.language_to_idx[language]
        
        return waveform, file_path, torch.tensor(language_idx, dtype=torch.long)


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

        print("  Wav2Vec2 model loaded and frozen")
        print(f"  Using layers {self.layer_indices} with learned weight aggregation")

    def forward(self, audio_data):
        """Extract and aggregate features from layers 17-24."""
        feat = self.processor(
            audio_data, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(audio_data.device)

        if feat.dim() == 3:
            feat = feat.squeeze(0)

        batch_size = feat.size(0)

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
    """Simple Projection Head for Language Identification.
    
    Architecture:
        Mean Pooling
            ↓
        Linear(1024 -> hidden_dim) + LayerNorm + GELU + Dropout
            ↓
        Linear(hidden_dim -> embedding_dim) + LayerNorm
    """

    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        embedding_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Simple projection head
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self._init_weights()

        print("  Simple Projection Head initialized:")
        print(f"    Input: {input_dim}D")
        print(f"    Hidden layer: {hidden_dim}D")
        print(f"    Output embedding: {embedding_dim}D")

    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

        print("\nBuilding Language ID Model:")
        print(f"  Number of languages: {num_languages}")

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
            labels: (B,) language class labels (required for training)

        Returns:
            emb_norm: normalized embeddings (B, embedding_dim)
            emb_unnorm: unnormalized embeddings (B, embedding_dim)
            logits: language logits (B, num_languages)
            loss: ArcFace loss (if labels provided, else None)
        """
        # Extract features
        feat = self.wav2vec_layer24(waveforms)

        # Get embeddings
        emb_unnorm = self.head(feat, normalize=False)
        emb_norm = self.head(feat, normalize=True)

        # Get logits and loss
        if labels is not None:
            loss, logits = self.arcface(emb_norm, labels)
            return emb_norm, emb_unnorm, logits, loss
        else:
            logits = self.arcface.forward_inference(emb_norm)
            return emb_norm, emb_unnorm, logits


def compute_eer_from_scores(y_true, y_score):
    """Compute Equal Error Rate (EER) from labels and scores."""
    fpr, fnr, threshold = compute_error_rates(y_true, y_score)
    
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold_val = threshold[eer_idx]
    
    return eer, threshold_val


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


def validate_language_recognition(model, val_loader, device, num_samples=5000):
    """
    Validate using language recognition EER on flag=2 data.
    Creates trial pairs from flag=2 samples: same language (target) vs different language (nontarget).
    
    Args:
        model: Language ID model
        val_loader: Validation DataLoader (flag=2)
        device: torch device
        num_samples: Number of trial pairs to generate
        
    Returns:
        eer: Equal Error Rate
        threshold: EER threshold
    """
    print("  Language Recognition (flag=2 EER)...")
    
    # Extract embeddings for all validation samples
    model.eval()
    embeddings_list = []
    languages_list = []
    
    with torch.no_grad():
        for waveform, filename, language_labels in val_loader:
            waveform = waveform.to(device)
            emb_norm, _, _ = model(waveform)
            embeddings_list.append(emb_norm.cpu().numpy())
            languages_list.extend(language_labels.numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0)  # (N, embedding_dim)
    languages = np.array(languages_list)  # (N,)
    
    # Group embeddings by language
    language_embeddings = {}
    for lang_idx in np.unique(languages):
        mask = languages == lang_idx
        language_embeddings[lang_idx] = embeddings[mask]
    
    # Generate trial pairs: same language (target=1) vs different language (nontarget=0)
    labels = []
    scores = []
    num_languages = len(language_embeddings)
    
    # Generate target pairs (same language)
    for lang_idx, embs in language_embeddings.items():
        if len(embs) >= 2:
            # Random pairs within same language
            n_pairs = min(num_samples // (2 * num_languages), len(embs) * (len(embs) - 1) // 2)
            
            for _ in range(n_pairs):
                idx1, idx2 = np.random.choice(len(embs), 2, replace=False)
                score = np.dot(embs[idx1], embs[idx2]) / (np.linalg.norm(embs[idx1]) * np.linalg.norm(embs[idx2]) + 1e-8)
                labels.append(1)  # target (same language)
                scores.append(score)
    
    # Generate non-target pairs (different languages)
    lang_indices = list(language_embeddings.keys())
    for i in range(len(lang_indices)):
        for j in range(i + 1, len(lang_indices)):
            lang_i, lang_j = lang_indices[i], lang_indices[j]
            embs_i = language_embeddings[lang_i]
            embs_j = language_embeddings[lang_j]
            
            # Random pairs between different languages
            n_pairs = min(num_samples // (2 * (num_languages * (num_languages - 1) // 2)), 
                         min(len(embs_i), len(embs_j)))
            
            for _ in range(n_pairs):
                idx1 = np.random.randint(0, len(embs_i))
                idx2 = np.random.randint(0, len(embs_j))
                score = np.dot(embs_i[idx1], embs_j[idx2]) / (np.linalg.norm(embs_i[idx1]) * np.linalg.norm(embs_j[idx2]) + 1e-8)
                labels.append(0)  # nontarget (different language)
                scores.append(score)
    
    # Compute EER
    labels = np.array(labels)
    scores = np.array(scores)
    
    if len(labels) == 0:
        print(f"    Could not generate trial pairs")
        return 100.0, 0.0
    
    eer, threshold = compute_eer_from_scores(labels, scores)
    
    # Statistics
    target_scores = scores[labels == 1]
    nontarget_scores = scores[labels == 0]
    
    if len(target_scores) > 0 and len(nontarget_scores) > 0:
        print(f"    Target (same language): mean={target_scores.mean():.4f}, std={target_scores.std():.4f}")
        print(f"    Non-target (diff language): mean={nontarget_scores.mean():.4f}, std={nontarget_scores.std():.4f}")
        print(f"    Gap: {target_scores.mean() - nontarget_scores.mean():.4f}")
    
    print(f"    EER: {100 * eer:.2f}% (threshold: {threshold:.6f})")
    
    return eer, threshold


def main():
    parser = argparse.ArgumentParser(description="Language Identification Training")
    
    # Model arguments
    parser.add_argument(
        "--ssl_model",
        type=str,
        default="facebook/wav2vec2-large",
        help="SSL model name (facebook/wav2vec2-large, facebook/xlsr-53-speak...)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of projection head",
    )
    # Data arguments
    parser.add_argument(
        "--unified_manifest",
        type=str,
        required=True,
        help="Path to unified manifest file (with flags)",
    )
    parser.add_argument(
        "--dataset_roots",
        type=str,
        nargs="+",
        required=True,
        help="Dataset root directories",
    )
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    
    # ArcFace arguments
    parser.add_argument(
        "--arcface_margin",
        type=float,
        default=0.3,
        help="ArcFace margin",
    )
    parser.add_argument(
        "--arcface_scale",
        type=float,
        default=30.0,
        help="ArcFace scale",
    )
    
    
    # Other arguments
    parser.add_argument(
        "--variance_weight",
        type=float,
        default=0.005,
        help="Weight for variance regularization loss",
    )
    parser.add_argument(
        "--out_fold",
        type=str,
        default="./output",
        help="Output folder",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=7,
        help="GPU ID",
    )
    
    args = parser.parse_args()

    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create output directory
    os.makedirs(args.out_fold, exist_ok=True)

    # Load datasets
    print("\nLoading training data (flag=1)...")
    train_dataset = UnifiedLanguageDataset(
        args.unified_manifest,
        args.dataset_roots,
        split_flag=1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print("\nLoading validation data (flag=2)...")
    val_dataset = UnifiedLanguageDataset(
        args.unified_manifest,
        args.dataset_roots,
        split_flag=2,
        language_to_idx=train_dataset.language_to_idx,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("\nLoading crosslingual validation data (flag=3)...")
    val_cl_dataset = UnifiedLanguageDataset(
        args.unified_manifest,
        args.dataset_roots,
        split_flag=3,
        language_to_idx=train_dataset.language_to_idx,
    )
    val_cl_loader = DataLoader(
        val_cl_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Verify all datasets have same language set
    assert train_dataset.language_to_idx == val_dataset.language_to_idx, \
        "Training and validation languages don't match!"
    assert train_dataset.language_to_idx == val_cl_dataset.language_to_idx, \
        "Training and crosslingual validation languages don't match!"

    # Build model
    print("\nBuilding model...")
    num_languages = len(train_dataset.language_to_idx)
    model = LanguageIDModel(
        num_languages=num_languages,
        ssl_model=args.ssl_model,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        device=device,
    )

    # Save language mapping
    with open(os.path.join(args.out_fold, "language_mapping.json"), "w") as f:
        json.dump(train_dataset.idx_to_language, f, indent=2)

    # Save arguments
    with open(os.path.join(args.out_fold, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    train_losses = []
    val_accs = []
    val_cl_accs = []
    lang_rec_eers = []
    best_macro_acc = 0.0

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        with torch.no_grad():
            weights = F.softmax(model.wav2vec_layer24.layer_weights, dim=0)
            weight_str = ", ".join(
                [f"L{l}={w:.3f}" for l, w in zip(range(17, 25), weights)]
            )

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for waveform, filename, language_labels in pbar:
            waveform = waveform.to(device)
            language_labels = language_labels.to(device)

            optimizer.zero_grad()

            emb_norm, emb_unnorm, logits, cls_loss = model(
                waveform, labels=language_labels
            )

            emb_var = emb_unnorm.var(dim=0).mean()
            variance_loss = torch.clamp(0.5 - emb_var, min=0.0)

            loss = cls_loss + args.variance_weight * variance_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += cls_loss.item()
            _, predicted = torch.max(logits, 1)
            total += language_labels.size(0)
            correct += (predicted == language_labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{cls_loss.item():.4f}",
                    "var_loss": f"{variance_loss.item():.4f}",
                    "emb_var": f"{emb_var.item():.4f}",
                    "acc": f"{100*correct/total:.1f}%",
                }
            )

        avg_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(avg_loss)

        print(f"  Train: loss={avg_loss:.4f}, acc={train_acc:.1f}%")

        # Validation (flag=2)
        print("  Validation (flag=2: new speakers)...")
        model.eval()
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels_list = []

        with torch.no_grad():
            for waveform, filename, language_labels in val_loader:
                waveform = waveform.to(device)
                language_labels = language_labels.to(device)

                emb_norm, emb_unnorm, logits = model(waveform)

                _, predicted = torch.max(logits, 1)
                val_total += language_labels.size(0)
                val_correct += (predicted == language_labels).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(language_labels.cpu().numpy())

        val_acc_micro = 100 * val_correct / val_total
        
        # Calculate macro-averaged accuracy: per-class accuracy averaged
        cm = confusion_matrix(val_labels_list, val_predictions, labels=list(range(num_languages)))
        per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-10)  # Avoid division by zero
        val_acc_macro = 100 * per_class_acc.mean()
        
        val_accs.append((val_acc_micro, val_acc_macro))

        print(f"  Val (flag=2): Micro={val_acc_micro:.1f}%, Macro={val_acc_macro:.1f}%")

        # Validation Crosslingual (flag=3)
        print("  Validation (flag=3: crosslingual)...")
        val_cl_correct = 0
        val_cl_total = 0
        val_cl_predictions = []
        val_cl_labels_list = []

        with torch.no_grad():
            for waveform, filename, language_labels in val_cl_loader:
                waveform = waveform.to(device)
                language_labels = language_labels.to(device)

                emb_norm, emb_unnorm, logits = model(waveform)

                _, predicted = torch.max(logits, 1)
                val_cl_total += language_labels.size(0)
                val_cl_correct += (predicted == language_labels).sum().item()
                
                val_cl_predictions.extend(predicted.cpu().numpy())
                val_cl_labels_list.extend(language_labels.cpu().numpy())

        val_cl_acc_micro = 100 * val_cl_correct / val_cl_total
        
        # Calculate macro-averaged accuracy: per-class accuracy averaged
        # Only compute over languages that actually appear in flag 3 (20 languages)
        unique_langs_cl = sorted(set(val_cl_labels_list))
        cm_cl = confusion_matrix(val_cl_labels_list, val_cl_predictions, labels=unique_langs_cl)
        per_class_acc_cl = np.diag(cm_cl) / (cm_cl.sum(axis=1) + 1e-10)  # Avoid division by zero
        val_cl_acc_macro = 100 * per_class_acc_cl.mean()
        
        val_cl_accs.append((val_cl_acc_micro, val_cl_acc_macro))

        print(f"  Val (flag=3): Micro={val_cl_acc_micro:.1f}%, Macro={val_cl_acc_macro:.1f}%")

        # Language Recognition (flag=2 EER)
        lang_rec_eer, lang_rec_threshold = validate_language_recognition(model, val_loader, device)
        lang_rec_eers.append(lang_rec_eer)

        # Save best checkpoint based on macro-averaged accuracy (flag=2)
        if val_acc_macro > best_macro_acc:
            best_macro_acc = val_acc_macro
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_acc_micro": val_acc_micro,
                    "val_acc_macro": val_acc_macro,
                    "val_cl_acc_micro": val_cl_acc_micro,
                    "val_cl_acc_macro": val_cl_acc_macro,
                    "layer_weights": model.wav2vec_layer24.layer_weights.detach().cpu(),
                },
                os.path.join(args.out_fold, "best_checkpoint.pt"),
            )
            print(f"  ✓ Best checkpoint saved! (Val Macro: {val_acc_macro:.1f}%)")

        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(args.out_fold, f"epoch_{epoch + 1:03d}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc_micro": val_acc_micro,
                "val_acc_macro": val_acc_macro,
                "val_cl_acc_micro": val_cl_acc_micro,
                "val_cl_acc_macro": val_cl_acc_macro,
                "layer_weights": model.wav2vec_layer24.layer_weights.detach().cpu(),
            },
            epoch_checkpoint_path,
        )

        # Log results
        with open(os.path.join(args.out_fold, "val_acc.log"), "w") as f:
            for i, (micro, macro) in enumerate(val_accs):
                f.write(f"Epoch {i}\tMicro: {micro:.2f}%\tMacro: {macro:.2f}%\n")

        with open(os.path.join(args.out_fold, "val_crosslingual_acc.log"), "w") as f:
            for i, (micro, macro) in enumerate(val_cl_accs):
                f.write(f"Epoch {i}\tMicro: {micro:.2f}%\tMacro: {macro:.2f}%\n")

        with open(os.path.join(args.out_fold, "lang_recognition_eer.log"), "w") as f:
            for i, eer in enumerate(lang_rec_eers):
                f.write(f"Epoch {i}\tEER: {100*eer:.2f}%\n")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation macro-averaged accuracy: {best_macro_acc:.1f}%")
    print()

    with torch.no_grad():
        weights = F.softmax(model.wav2vec_layer24.layer_weights, dim=0)
        print("Final layer weights (layers 17-24):")
        for layer_id, weight in zip(range(17, 25), weights):
            print(f"  Layer {layer_id}: {weight:.4f}")
        print()


if __name__ == "__main__":
    main()
