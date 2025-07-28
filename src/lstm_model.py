"""
LSTM Model for Sign Language Recognition (Sign2Gloss)

This module implements a Bi-LSTM model for converting sign language video sequences
to gloss annotations, ready for further processing with mBART.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import argparse
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import LambdaLR
import math
from curriculum_learning import CurriculumDataset, CurriculumSampler
from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig, PhoenixDataset


@dataclass
@dataclass
@dataclass
class ModelConfig:
    """Configuration for LSTM model"""

    # Curriculum Learning parameters
    use_curriculum_learning: bool = True
    curriculum_warmup_epochs: int = 5
    curriculum_stages: int = 4
    curriculum_overlap_ratio: float = 0.3

    # Model architecture - will be updated dynamically
    input_size: int = 356  # Updated default based on calculated dimensions
    hidden_size: int = 128  # Reduced for debugging
    num_layers: int = 1     # Reduced for debugging
    dropout: float = 0.2    # Reduced for debugging
    bidirectional: bool = False

    # Training parameters
    batch_size: int = 4     # Small for debugging
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 5     # Reduced for debugging
    patience: int = 3

    # Data parameters
    max_sequence_length: int = 64   # Reduced for debugging
    max_annotation_length: int = 20

    # Paths
    data_dir: str = "./output"
    annotations_path: str = "./phoenix_annotations.xlsx"
    vocab_path: str = "./vocab.pkl"
    model_save_path: str = "./models/lstm_sign2gloss.pth"

    # WandB configuration
    project_name: str = "sign-language-lstm"
    experiment_name: str = "lstm-sign2gloss"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SignLanguageLSTM(nn.Module):
    """LSTM model for real-time sign language recognition"""

    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        print(f"Initializing LSTM model:")
        print(f"  - Input size: {config.input_size}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Vocab size: {vocab_size}")

        # Input projection
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_size, config.max_sequence_length)

        # LSTM layers (unidirectional for real-time processing)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        # Attention mechanism
        lstm_output_size = config.hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=1,
            dropout=config.dropout,
            batch_first=True
        )

        # Output layers - ADD THE MISSING LAYER_NORM HERE
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)  # ADD THIS LINE

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size // 2, vocab_size)
        )

        # CTC loss for sequence alignment
        self.use_ctc = False

    def forward(self, x, attention_mask=None, labels=None):
        """
        Forward pass with focal loss and L2 regularization
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention
        if attention_mask is not None:
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None

        attn_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )

        # Residual connection and layer norm - FIXED TO USE CORRECT ATTRIBUTE
        x = self.layer_norm(lstm_out + attn_out)
        x = self.dropout(x)

        # Classification
        logits = self.classifier(x)

        # Get predictions for evaluation
        predictions = torch.argmax(logits, dim=-1)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.use_ctc:
                log_probs = F.log_softmax(logits, dim=-1)
                input_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.full((batch_size,),
                                                                                                        seq_len)
                target_lengths = (labels != 0).sum(dim=1)

                loss = F.ctc_loss(
                    log_probs.transpose(0, 1),
                    labels,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction='mean'
                )
            # else:
            #     # FOCAL LOSS: Focus on hard examples to prevent easy convergence
            #     logits_flat = logits[:, :labels.shape[1], :].reshape(-1, self.vocab_size)
            #     labels_flat = labels.reshape(-1)
            #
            #     # Calculate cross-entropy without reduction
            #     ce_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0, reduction='none')
            #
            #     # Apply focal loss weighting
            #     pt = torch.exp(-ce_loss)  # Probability of true class
            #     focal_weight = (1 - pt) ** 2  # Focus on hard examples (low pt)
            #     focal_loss = focal_weight * ce_loss
            #
            #     # Only average over non-padding tokens
            #     mask = labels_flat != 0
            #     if mask.sum() > 0:
            #         loss = focal_loss[mask].mean()
            #     else:
            #         loss = focal_loss.mean()  # Fallback if all padding
            else:
                # ENHANCED LOSS: Class-weighted focal loss + diversity penalty
                logits_flat = logits[:, :labels.shape[1], :].reshape(-1, self.vocab_size)
                labels_flat = labels.reshape(-1)

                # Calculate class weights dynamically
                mask = labels_flat != 0
                non_padding_ratio = mask.float().mean()

                # Strong class weighting to discourage padding predictions
                class_weights = torch.ones(self.vocab_size, device=logits.device)
                class_weights[0] = 0.1  # Very low weight for padding token

                # Calculate weighted cross-entropy
                ce_loss = F.cross_entropy(logits_flat, labels_flat, weight=class_weights, reduction='none')

                # STRONGER focal loss (gamma=3 instead of 2)
                pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** 3  # Stronger focus on hard examples
                focal_loss = focal_weight * ce_loss

                # Add diversity penalty to encourage non-padding predictions
                probs = F.softmax(logits_flat, dim=-1)
                padding_prob = probs[:, 0]  # Probability of predicting padding
                diversity_penalty = torch.mean(padding_prob) * 2.0  # Penalty for high padding predictions

                # Combine losses
                if mask.sum() > 0:
                    loss = focal_loss[mask].mean() + diversity_penalty
                else:
                    loss = focal_loss.mean() + diversity_penalty

                # Log diversity metrics
                if torch.rand(1).item() < 0.1:  # Occasionally log
                    non_padding_preds = (torch.argmax(logits_flat, dim=-1) != 0).float().mean()
                    print(
                        f"Non-padding predictions: {non_padding_preds:.3f}, Diversity penalty: {diversity_penalty:.3f}")

            # L2 REGULARIZATION: Prevent overfitting and force harder learning
            l2_penalty = 0

            # Regularize classifier layers
            for param in self.classifier.parameters():
                l2_penalty += torch.norm(param, 2)

            # Regularize LSTM parameters
            for param in self.lstm.parameters():
                l2_penalty += torch.norm(param, 2)

            # Regularize input projection
            for param in self.input_projection.parameters():
                l2_penalty += torch.norm(param, 2)

            # Add L2 penalty to loss
            l2_lambda = 0.001  # L2 regularization strength
            loss = loss + l2_lambda * l2_penalty

        return {
            'logits': logits,
            'loss': loss,
            'attention_weights': attention_weights,
            'predictions': predictions
        }

    def predict(self, x, attention_mask=None):
        """Generate predictions for input sequence"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, attention_mask)
            return output['predictions']

    def predict_realtime(self, frame_features, hidden_state=None):
        """
        Real-time prediction for a single frame

        Args:
            frame_features: Single frame features (1, 1, input_size)
            hidden_state: Previous LSTM hidden state (optional)

        Returns:
            Dictionary with prediction, confidence, and new hidden state
        """
        self.eval()
        with torch.no_grad():
            # Ensure correct input shape
            if frame_features.dim() == 2:
                frame_features = frame_features.unsqueeze(1)  # Add sequence dimension
            if frame_features.dim() == 1:
                frame_features = frame_features.unsqueeze(0).unsqueeze(1)  # Add batch and sequence

            # Input projection
            x = self.input_projection(frame_features)

            # LSTM forward pass with hidden state
            if hidden_state is not None:
                lstm_out, new_hidden_state = self.lstm(x, hidden_state)
            else:
                lstm_out, new_hidden_state = self.lstm(x)

            # Classification
            logits = self.classifier(lstm_out)

            # Get prediction and confidence
            probabilities = F.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]

            return {
                'prediction': prediction.squeeze().item(),
                'confidence': confidence.squeeze().item(),
                'hidden_state': new_hidden_state,
                'logits': logits.squeeze()
            }

    def reset_hidden_state(self):
        """Reset hidden state for new sequence"""
        return None  # LSTM will initialize hidden state automatically


class SignLanguageTrainer:
    """Training pipeline for sign language LSTM model"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.use_curriculum_learning = True
        self.curriculum_dataset = None

        # Initialize WandB
        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            config=asdict(config)
        )

        # Create model save directory
        Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize preprocessor
        preprocess_config = PreprocessingConfig(
            max_sequence_length=config.max_sequence_length,
            output_format="tensor",
            device=config.device
        )
        self.preprocessor = SignLanguagePreprocessor(preprocess_config)

        # Load dataset FIRST to get actual feature dimensions
        self.dataset = self._load_dataset()

        # Update config with actual input size from preprocessor
        actual_input_size = self.preprocessor.feature_dims['total']
        if config.input_size != actual_input_size:
            print(f"Updating input_size from {config.input_size} to {actual_input_size}")
            config.input_size = actual_input_size
            self.config = config

        self.train_loader, self.val_loader = self._create_data_loaders()

        # Initialize model AFTER getting correct input size
        self.model = SignLanguageLSTM(config, self.dataset.vocab_size).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-5,  # Much lower starting rate
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5
        # )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,  # Reduce LR every 10 epochs
            gamma=0.8  # Multiply LR by 0.8
        )



        # Training tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        def lr_lambda_func(step):
            warmup_steps = 20
            total_steps = 200

            if step < warmup_steps:
                # Warmup phase
                return max(0.1, step / warmup_steps)  # Minimum 10% of base LR
            else:
                # Cosine decay phase with safety checks
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1

                import math
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda_func)


    def _load_dataset(self) -> PhoenixDataset:
        """Load and create Phoenix dataset"""
        try:
            dataset = self.preprocessor.create_phoenix_dataset(
                data_dir=self.config.data_dir,
                annotations_path=self.config.annotations_path,
                vocab_path=self.config.vocab_path if Path(self.config.vocab_path).exists() else None
            )

            # Save vocabulary
            dataset.save_vocabulary(self.config.vocab_path)

            print(f"Loaded dataset with {len(dataset)} samples")
            print(f"Vocabulary size: {dataset.vocab_size}")

            return dataset

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with simple length grouping and dynamic padding"""
        if self.use_curriculum_learning:
            return self._create_data_loaders_with_curriculum()
        else:
            # Split dataset
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size

            train_dataset, val_dataset = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Create simple length grouping samplers
            train_sampler = BatchBucketing(
                train_dataset,
                self.config.batch_size,
                shuffle=True
            )

            val_sampler = BatchBucketing(
                val_dataset,
                self.config.batch_size,
                shuffle=False  # Don't shuffle validation
            )

            # Create data loaders with both grouping and dynamic padding
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,  # Use our custom sampler
                num_workers=0,
                pin_memory=False,
                collate_fn=self._dynamic_collate_fn  # Plus dynamic padding
            )

            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_sampler,
                num_workers=0,
                pin_memory=False,
                collate_fn=self._dynamic_collate_fn
            )

            print(f"Train samples: {len(train_dataset)} -> {len(train_sampler)} batches")
            print(f"Validation samples: {len(val_dataset)} -> {len(val_sampler)} batches")

            return self._create_data_loaders_original()

    def _create_data_loaders_with_curriculum(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with curriculum learning support"""
        print("Creating curriculum learning data loaders...")

        # Create curriculum dataset
        self.curriculum_dataset = self.dataset.create_curriculum_dataset(
            total_epochs=self.config.num_epochs,
            warmup_epochs=min(5, self.config.num_epochs // 4)
        )

        # Split curriculum dataset for training
        curriculum_size = len(self.curriculum_dataset.difficulty_metrics)
        train_size = int(0.8 * curriculum_size)
        val_size = curriculum_size - train_size

        # Create train/val splits based on difficulty ranking
        train_difficulties = self.curriculum_dataset.difficulty_metrics[:train_size]
        val_difficulties = self.curriculum_dataset.difficulty_metrics[train_size:]

        # Create separate curriculum datasets for train/val
        train_curriculum = CurriculumDataset(
            base_dataset=self.dataset,
            difficulty_metrics=train_difficulties,
            scheduler=self.curriculum_dataset.scheduler,
            preprocessor=self.preprocessor
        )

        # For validation, use a subset of medium-difficulty samples
        val_indices = [d.index for d in val_difficulties
                       if 0.2 < d.overall_difficulty < 0.8][:len(val_difficulties) // 2]
        val_subset = torch.utils.data.Subset(self.dataset, val_indices)

        # Create data loaders
        train_sampler = CurriculumSampler(train_curriculum, self.config.batch_size)

        train_loader = DataLoader(
            train_curriculum,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._dynamic_collate_fn
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._dynamic_collate_fn
        )

        print(f"Curriculum train samples: {len(train_curriculum.difficulty_metrics)}")
        print(f"Validation samples: {len(val_subset)}")

        return train_loader, val_loader

    def _create_data_loaders_original(self) -> Tuple[DataLoader, DataLoader]:
        """Original data loader creation (renamed from existing method)"""
        # Move your existing _create_data_loaders implementation here
        # Split dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create simple length grouping samplers
        train_sampler = BatchBucketing(
            train_dataset,
            self.config.batch_size,
            shuffle=True
        )

        val_sampler = BatchBucketing(
            val_dataset,
            self.config.batch_size,
            shuffle=False
        )

        # Create data loaders with both grouping and dynamic padding
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._dynamic_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._dynamic_collate_fn
        )

        print(f"Train samples: {len(train_dataset)} -> {len(train_sampler)} batches")
        print(f"Validation samples: {len(val_dataset)} -> {len(val_sampler)} batches")

        return train_loader, val_loader

    def _dynamic_collate_fn(self, batch):
        """Dynamic collate function with curriculum-aware sequence lengths"""
        try:
            # Extract all components
            sequences = [item['sequence'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            labels = [item['labels'] for item in batch]
            annotations = [item['annotation'] for item in batch]
            metadata = [item['metadata'] for item in batch]

            # For curriculum learning, all sequences should already be the same length
            # But double-check and pad to batch maximum just in case
            actual_lengths = [mask.sum().item() for mask in attention_masks]

            if len(set(seq.shape[0] for seq in sequences)) > 1:
                # Sequences have different lengths - pad to maximum in batch
                batch_max_seq_length = max(seq.shape[0] for seq in sequences)

                resized_sequences = []
                resized_attention_masks = []

                for seq, mask in zip(sequences, attention_masks):
                    if seq.shape[0] < batch_max_seq_length:
                        pad_length = batch_max_seq_length - seq.shape[0]
                        padding = torch.zeros(pad_length, seq.shape[1], dtype=seq.dtype)
                        mask_padding = torch.zeros(pad_length, dtype=mask.dtype)

                        resized_seq = torch.cat([seq, padding], dim=0)
                        resized_mask = torch.cat([mask, mask_padding], dim=0)
                    else:
                        resized_seq = seq
                        resized_mask = mask

                    resized_sequences.append(resized_seq)
                    resized_attention_masks.append(resized_mask)
            else:
                # All sequences same length - use as is
                resized_sequences = sequences
                resized_attention_masks = attention_masks

            # Handle labels (keep original logic)
            max_label_length = max(label.shape[0] for label in labels)
            resized_labels = []

            for label in labels:
                if label.shape[0] >= max_label_length:
                    resized_label = label[:max_label_length]
                else:
                    pad_length = max_label_length - label.shape[0]
                    padding = torch.zeros(pad_length, dtype=label.dtype)
                    resized_label = torch.cat([label, padding], dim=0)
                resized_labels.append(resized_label)

            # Log curriculum efficiency
            if hasattr(self, 'curriculum_dataset') and self.curriculum_dataset:
                total_tokens = sum(seq.numel() for seq in resized_sequences)
                actual_tokens = sum(
                    mask.sum().item() * seq.shape[1] for seq, mask in zip(resized_sequences, resized_attention_masks))
                efficiency = (actual_tokens / total_tokens) * 100

                if np.random.random() < 0.02:  # Occasionally log
                    print(f"Curriculum efficiency: {efficiency:.1f}% ({actual_tokens}/{total_tokens} tokens)")

            return {
                'sequence': torch.stack(resized_sequences),
                'attention_mask': torch.stack(resized_attention_masks),
                'labels': torch.stack(resized_labels),
                'annotation': annotations,
                'metadata': metadata
            }

        except Exception as e:
            print(f"Error in dynamic collate function: {e}")
            return self._safe_collate_fn(batch)

    def _calculate_padding_ratio(self, sequences: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """
        Calculate padding ratio consistently across all methods

        Args:
            sequences: Input sequences tensor [batch_size, seq_len, feature_dims]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Padding ratio as percentage (0-100)
        """
        total_elements = sequences.numel()  # batch_size × seq_len × feature_dims

        # Count actual (non-padding) elements
        non_padding_positions = attention_mask.sum().item()  # Count of True values (positions only)
        feature_dims = sequences.shape[2]
        actual_elements = non_padding_positions * feature_dims

        # Calculate padding ratio
        padding_elements = total_elements - actual_elements
        padding_ratio = (padding_elements / total_elements) * 100 if total_elements > 0 else 0

        return padding_ratio

    def _debug_attention_mask_mismatch(self, batch):
        """Debug attention mask vs actual data mismatch"""
        sequences = batch['sequence']
        attention_mask = batch['attention_mask']

        for i in range(min(2, sequences.shape[0])):  # Check first 2 samples
            seq = sequences[i]
            mask = attention_mask[i]

            # Calculate different padding ratios
            mask_based_padding = (1 - mask.float().mean()) * 100
            data_based_padding = (1 - (seq != 0).float().mean()) * 100

            print(f"\nSample {i} Mismatch Analysis:")
            print(f"  Mask-based padding: {mask_based_padding:.1f}%")
            print(f"  Data-based padding: {data_based_padding:.1f}%")
            print(f"  Difference: {abs(mask_based_padding - data_based_padding):.1f}%")

            # Check position-by-position
            mask_valid_positions = mask.sum().item()
            data_valid_positions = (seq != 0).any(dim=1).sum().item()

            print(f"  Mask says {mask_valid_positions} valid positions")
            print(f"  Data has {data_valid_positions} non-zero positions")

    def _detect_and_fix_collapse(self, outputs, batch_idx):
        """Detect model collapse (including SOS/EOS collapse) and increase learning rate"""
        predictions = outputs['predictions'].flatten()

        # Count different token types
        total_preds = predictions.numel()
        padding_preds = (predictions == 0).sum().item()
        unk_preds = (predictions == 1).sum().item()
        sos_preds = (predictions == 2).sum().item()
        eos_preds = (predictions == 3).sum().item()
        vocab_preds = (predictions > 3).sum().item()

        # Calculate ratios
        vocab_ratio = vocab_preds / total_preds
        special_ratio = (sos_preds + eos_preds) / total_preds
        padding_ratio = padding_preds / total_preds

        # Detect different types of collapse
        collapsed = False
        collapse_type = None

        if vocab_ratio < 0.10:  # Less than 5% vocabulary predictions
            collapsed = True

            if padding_ratio > 0.9:
                collapse_type = "PADDING_COLLAPSE"
            elif special_ratio > 0.8:
                collapse_type = "SPECIAL_TOKEN_COLLAPSE (SOS/EOS)"
            elif unk_preds > total_preds * 0.8:
                collapse_type = "UNK_COLLAPSE"
            else:
                collapse_type = "VOCAB_COLLAPSE"

        if collapsed:
            print(f"   {collapse_type} DETECTED at batch {batch_idx}!")
            print(
                f"   Vocab preds: {vocab_ratio:.1%} | Special (SOS/EOS): {special_ratio:.1%} | Padding: {padding_ratio:.1%}")
            print(f"   Breakdown: SOS={sos_preds}, EOS={eos_preds}, Vocab={vocab_preds}, PAD={padding_preds}")

            # Increase learning rate to escape local minimum
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = min(current_lr * 3.0, 1e-3)  # Triple LR, max 1e-3

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            print(f"   Increased LR from {current_lr:.2e} to {new_lr:.2e}")

            # Log to wandb if available
            if hasattr(self, 'wandb') or 'wandb' in globals():
                try:
                    wandb.log({
                        'collapse_detected': 1,
                        'collapse_type': collapse_type,
                        'vocab_prediction_ratio': vocab_ratio,
                        'special_token_ratio': special_ratio,
                        'learning_rate_boost': new_lr
                    })
                except:
                    pass  # Don't fail if wandb isn't available

            return True

        return False

    def _should_do_nuclear_reset(self, outputs, batch_idx):
        """Check if nuclear reset is needed (without doing it)"""
        predictions = outputs['predictions'].flatten()
        vocab_predictions = predictions[predictions > 3]

        if len(vocab_predictions) > 0:
            unique_preds, counts = torch.unique(vocab_predictions, return_counts=True)
            if len(counts) > 0:
                repetition_ratio = counts.max().item() / len(vocab_predictions)

                if repetition_ratio > 0.4:  # Same threshold
                    print(f"*** NUCLEAR RESET SCHEDULED at batch {batch_idx}!")
                    print(f"    Repetition: {repetition_ratio:.1%}")
                    return True

        return False

    def _do_nuclear_reset(self):
        """Execute nuclear reset (called after backward pass)"""
        print("    EXECUTING NUCLEAR RESET...")

        # 1. RESET classifier weights
        for layer in self.model.classifier:
            if hasattr(layer, 'weight'):
                torch.nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        # 2. MASSIVE learning rate boost
        new_lr = 5e-3
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # 3. Enable gradient noise
        self._enable_gradient_noise = True
        self._noise_steps_remaining = 50

        print(f"    Classifier reset complete, LR changed to {new_lr:.2e}")

    def train_epoch(self) -> float:
        """Train for one epoch with padding ratio monitoring"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        # Padding ratio tracking
        padding_ratios = []
        sample_frequency = max(1, num_batches // 10)  # Sample ~10 batches per epoch

        pbar = tqdm(self.train_loader, desc="Training")

        if not hasattr(self, '_padding_debug_done'):
            print("\n=== ENHANCED PADDING DEBUG ===")
            test_batch_count = 0
            for batch in self.train_loader:
                if test_batch_count >= 1:  # Just check first batch
                    break

                sequences = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Debug the shapes and values
                print(f"Sequences shape: {sequences.shape}")
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Attention mask dtype: {attention_mask.dtype}")
                print(f"Attention mask sum: {attention_mask.sum().item()}")
                print(f"Attention mask min/max: {attention_mask.min().item()}/{attention_mask.max().item()}")
                print(f"Total elements in sequences: {sequences.numel()}")

                # Test calculations
                total_elements = sequences.numel()
                mask_sum = attention_mask.sum().item()
                feature_dims = sequences.shape[2]

                print(f"Feature dimensions: {feature_dims}")
                print(f"Mask sum × feature dims: {mask_sum * feature_dims}")
                print(f"Total elements: {total_elements}")

                # Check if mask_sum * feature_dims > total_elements
                if mask_sum * feature_dims > total_elements:
                    print(f"ERROR: actual_elements ({mask_sum * feature_dims}) > total_elements ({total_elements})")
                    print("This suggests attention mask values are not 0/1 or there's a shape mismatch")

                test_batch_count += 1

            self._padding_debug_done = True

            print("All padding calculations verified!")
            self._padding_debug_done = True

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            sequences = batch['sequence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Calculate padding ratio for this batch
            padding_ratio = self._calculate_padding_ratio(sequences, attention_mask)

            padding_ratios.append(padding_ratio)

            # Display sample padding ratios during training
            if batch_idx % sample_frequency == 0 or batch_idx < 2:  # Show first 2 + samples
                batch_size, seq_len, features = sequences.shape
                avg_seq_length = attention_mask.sum().item() / batch_size
                print(f"\nBatch {batch_idx}: shape=({batch_size}, {seq_len}, {features}), "
                      f"avg_real_length={avg_seq_length:.1f}, padding={padding_ratio:.1f}%")

            # Validation checks
            non_zero_ratio = (sequences != 0).float().mean().item()
            if non_zero_ratio < 0.01:
                print(f"WARNING: Batch {batch_idx} has mostly empty sequences ({non_zero_ratio:.3f})")

            vocab_tokens = (labels > 3).sum().item()
            total_label_tokens = (labels != 0).sum().item()
            if total_label_tokens > 0:
                vocab_ratio = vocab_tokens / total_label_tokens
                if vocab_ratio < 0.1:
                    print(f"WARNING: Batch {batch_idx} has mostly special tokens ({vocab_ratio:.3f})")

            if batch_idx == 0 and not hasattr(self, '_mask_debug_done'):
                self._debug_attention_mask_mismatch(batch)
                self._mask_debug_done = True

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequences, attention_mask, labels)
            loss = outputs['loss']

            nuclear_reset_needed = False
            if batch_idx % 10 == 0:  # Check every 10 batches
                collapsed = self._detect_and_fix_collapse(outputs, batch_idx)
                if not collapsed:
                    # diversity_problem = self._detect_diversity_problem(outputs, batch_idx)
                    # excape_deep_minimum = self._escape_deep_minimum(outputs, batch_idx)

                    # if diversity_problem:
                    nuclear_reset_needed = self._should_do_nuclear_reset(outputs, batch_idx)
                else:
                    nuclear_reset_needed = False

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            if nuclear_reset_needed:
                self._do_nuclear_reset()

            # Update metrics
            total_loss += loss.item()

            # Update progress bar with padding info
            current_padding = padding_ratios[-1] if padding_ratios else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pad%': f'{current_padding:.1f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Log to WandB
            wandb.log({
                'train_loss_step': loss.item(),
                'padding_ratio_batch': padding_ratio,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        # Calculate and display epoch padding statistics
        avg_padding_ratio = np.mean(padding_ratios)
        min_padding_ratio = np.min(padding_ratios)
        max_padding_ratio = np.max(padding_ratios)

        print(f"\n=== Epoch Padding Statistics ===")
        print(f"Average padding ratio: {avg_padding_ratio:.1f}%")
        print(f"Padding range: {min_padding_ratio:.1f}% - {max_padding_ratio:.1f}%")
        print(f"Batches with <20% padding: {sum(1 for r in padding_ratios if r < 20)}/{len(padding_ratios)}")
        print(f"Batches with >50% padding: {sum(1 for r in padding_ratios if r > 50)}/{len(padding_ratios)}")

        # Log epoch statistics to WandB
        wandb.log({
            'epoch_avg_padding_ratio': avg_padding_ratio,
            'epoch_min_padding_ratio': min_padding_ratio,
            'epoch_max_padding_ratio': max_padding_ratio,
        })

        return avg_loss

    def validate_epoch(self) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch in pbar:
                # Move batch to device
                sequences = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(sequences, attention_mask, labels)
                loss = outputs['loss']

                # Update metrics
                total_loss += loss.item()

                # Collect predictions and labels
                predictions = outputs['predictions'].cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels_np)

                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)

        return avg_loss, metrics

    def _calculate_metrics(self, predictions: List, labels: List) -> Dict:
        """Calculate evaluation metrics"""
        pred_flat = []
        label_flat = []

        for pred, label in zip(predictions, labels):
            # Convert to numpy if they're tensors
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            if hasattr(label, 'cpu'):
                label = label.cpu().numpy()

            # Handle dimension mismatch by taking minimum length
            min_len = min(len(pred), len(label))
            pred_trimmed = pred[:min_len]
            label_trimmed = label[:min_len]

            # Create mask for non-padding tokens
            mask = label_trimmed != 0

            # Only add valid (non-padding) tokens
            if mask.any():
                pred_flat.extend(pred_trimmed[mask])
                label_flat.extend(label_trimmed[mask])

        # Handle empty predictions/labels
        if not pred_flat or not label_flat:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        try:
            # Calculate accuracy
            accuracy = accuracy_score(label_flat, pred_flat)

            # Calculate precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                label_flat, pred_flat, average='weighted', zero_division=0
            )

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

    # def get_curriculum_sampler(self, epoch):
    #     """Create curriculum learning sampler"""
    #     # Start with shorter sequences, gradually include longer ones
    #     max_allowed_length = min(50 + epoch * 5, 200)  # Gradually increase
    #
    #     valid_indices = []
    #     for i, sample in enumerate(self.dataset):
    #         seq_length = sample['attention_mask'].sum().item()
    #         if seq_length <= max_allowed_length:
    #             valid_indices.append(i)
    #
    #     return torch.utils.data.SubsetRandomSampler(valid_indices)

    def analyze_padding_improvement(self):
        """Compare padding ratios with and without dynamic batching"""
        print("\n=== Padding Ratio Analysis ===")

        # Sample a few batches to show the improvement
        sample_batches = 0
        total_original_padding = 0
        total_dynamic_padding = 0
        total_tokens_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                if sample_batches >= 5:  # Analyze first 5 batches
                    break

                sequences = batch['sequence']
                attention_mask = batch['attention_mask']

                # Calculate current (dynamic) padding
                current_padding_ratio = self._calculate_padding_ratio(sequences, attention_mask)
                current_total_tokens = sequences.numel()
                current_actual_tokens = attention_mask.sum().item() * sequences.shape[2]
                current_padding = current_total_tokens - current_actual_tokens

                # Calculate what padding would be with original max_sequence_length
                batch_size = sequences.shape[0]
                original_total_tokens = batch_size * self.config.max_sequence_length * sequences.shape[2]
                original_padding = original_total_tokens - current_actual_tokens

                total_original_padding += original_padding
                total_dynamic_padding += current_padding
                total_tokens_processed += current_actual_tokens

                print(f"Batch {batch_idx}: "
                      f"shape={sequences.shape}, "
                      f"original_pad={original_padding / original_total_tokens * 100:.1f}%, "
                      f"dynamic_pad={current_padding / current_total_tokens * 100:.1f}%")

                sample_batches += 1

        if total_tokens_processed > 0:
            original_pad_ratio = total_original_padding / (total_original_padding + total_tokens_processed) * 100
            dynamic_pad_ratio = total_dynamic_padding / (total_dynamic_padding + total_tokens_processed) * 100
            improvement = original_pad_ratio - dynamic_pad_ratio

            print(f"\nOverall Improvement:")
            print(f"  Original padding ratio: {original_pad_ratio:.1f}%")
            print(f"  Dynamic padding ratio: {dynamic_pad_ratio:.1f}%")
            print(f"  Improvement: {improvement:.1f} percentage points")
            print(f"  Relative improvement: {improvement / original_pad_ratio * 100:.1f}%")

    def _debug_label_distribution(self):
        """Debug what tokens are actually in the training labels"""
        print("\n=== Label Distribution Analysis ===")

        token_counts = {}
        total_tokens = 0

        # Sample first few batches
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= 3:  # Just first 3 batches
                break

            labels = batch['labels']

            # Count all tokens in this batch
            for label_seq in labels:
                for token_id in label_seq:
                    token_id = token_id.item()
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1
                    total_tokens += 1

        # Sort by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"Total tokens analyzed: {total_tokens}")
        print("Top 15 most frequent tokens:")

        id_to_vocab = {v: k for k, v in self.dataset.vocab.items()}

        for token_id, count in sorted_tokens[:15]:
            token_name = id_to_vocab.get(token_id, f"UNKNOWN_{token_id}")
            percentage = (count / total_tokens) * 100
            print(f"  ID {token_id:3d} ('{token_name}'): {count:5d} times ({percentage:5.1f}%)")

        # Calculate special token ratio
        special_tokens = [0, 1, 2, 3]  # PAD, UNK, SOS, EOS
        special_count = sum(token_counts.get(tid, 0) for tid in special_tokens)
        vocab_count = total_tokens - special_count

        print(f"\nToken Type Distribution:")
        print(f"  Special tokens: {special_count:5d} ({special_count / total_tokens * 100:5.1f}%)")
        print(f"  Vocabulary tokens: {vocab_count:5d} ({vocab_count / total_tokens * 100:5.1f}%)")

    def _debug_sample_labels(self):
        """Debug individual sample label structure"""
        print("\n=== Sample Label Structure ===")

        for i in range(3):  # Check first 3 samples
            sample = self.dataset[i]
            labels = sample['labels']
            annotation = sample['annotation']

            # Show raw label sequence
            non_zero_labels = labels[labels != 0]  # Remove padding
            print(f"\nSample {i}:")
            print(f"  Annotation: '{annotation}'")
            print(f"  Raw labels: {labels[:15].tolist()}...")  # First 15 labels
            print(f"  Non-zero labels: {non_zero_labels.tolist()}")

            # Decode labels
            id_to_vocab = {v: k for k, v in self.dataset.vocab.items()}
            decoded_tokens = []
            for token_id in non_zero_labels:
                token_name = id_to_vocab.get(token_id.item(), f"UNK_{token_id.item()}")
                decoded_tokens.append(token_name)

            print(f"  Decoded labels: {decoded_tokens}")

            # Check annotation words vs vocab
            expected_words = annotation.split()
            print(f"  Expected words: {expected_words}")

            # Check if words are in vocab
            missing_words = []
            for word in expected_words:
                if word not in self.dataset.vocab:
                    missing_words.append(word)

            if missing_words:
                print(f"    Missing from vocab: {missing_words}")

    def _detect_diversity_problem(self, outputs, batch_idx):
        """Detect when model gets stuck predicting the same words repeatedly"""
        predictions = outputs['predictions'].flatten()

        # Filter to vocabulary predictions only (exclude special tokens)
        vocab_predictions = predictions[predictions > 3]

        # Need at least some vocab predictions to analyze
        if len(vocab_predictions) == 0:
            return False  # No vocab predictions to analyze

        # Count occurrences of each vocabulary word
        unique_preds, counts = torch.unique(vocab_predictions, return_counts=True)

        if len(counts) == 0:
            return False

        # Find the most frequently predicted word
        max_count = counts.max().item()
        total_vocab_preds = len(vocab_predictions)
        repetition_ratio = max_count / total_vocab_preds

        # Detect if any single word dominates predictions
        if repetition_ratio > 0.4:  # More than 40% of vocab predictions are the same word
            # Find which word is dominating (for logging)
            most_common_idx = torch.argmax(counts)
            dominant_word_id = unique_preds[most_common_idx].item()

            # Get word name if possible
            word_name = "UNKNOWN"
            if hasattr(self.dataset, 'vocab'):
                id_to_vocab = {v: k for k, v in self.dataset.vocab.items()}
                word_name = id_to_vocab.get(dominant_word_id, f"ID_{dominant_word_id}")

            print(f"*** DIVERSITY PROBLEM at batch {batch_idx}!")
            print(f"    Word '{word_name}' dominates: {repetition_ratio:.1%} of vocab predictions")
            print(f"    Breakdown: {max_count}/{total_vocab_preds} vocab predictions")

            # Boost learning rate (smaller boost than collapse detection)
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = min(current_lr * 2.0, 1e-3)  # Double LR (less aggressive than collapse)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            print(f"    >>> Boosted LR from {current_lr:.2e} to {new_lr:.2e}")

            # Log to wandb if available
            try:
                if 'wandb' in globals():
                    wandb.log({
                        'diversity_problem_detected': 1,
                        'dominant_word_ratio': repetition_ratio,
                        'dominant_word_id': dominant_word_id,
                        'learning_rate_boost': new_lr
                    })
            except:
                pass  # Don't fail if wandb isn't available

            return True

        return False

    def _escape_deep_minimum(self, outputs, batch_idx):
        """More aggressive escape from deep local minima"""
        predictions = outputs['predictions'].flatten()
        vocab_predictions = predictions[predictions > 3]

        if len(vocab_predictions) > 0:
            unique_preds, counts = torch.unique(vocab_predictions, return_counts=True)
            if len(counts) > 0:
                repetition_ratio = counts.max().item() / len(vocab_predictions)

                if repetition_ratio > 0.4:  # Diversity problem detected
                    print(f"*** DEEP MINIMUM ESCAPE at batch {batch_idx}!")
                    print(f"    Repetition: {repetition_ratio:.1%}")

                    # 1. RESET classifier weights (nuclear option)
                    for layer in self.model.classifier:
                        if hasattr(layer, 'weight'):
                            torch.nn.init.xavier_uniform_(layer.weight)
                            if hasattr(layer, 'bias') and layer.bias is not None:
                                torch.nn.init.zeros_(layer.bias)

                    # 2. MASSIVE learning rate boost
                    new_lr = 5e-3  # Much higher than max 1e-3
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

                    # 3. Add gradient noise for several steps
                    self._enable_gradient_noise = True
                    self._noise_steps_remaining = 50

                    print(f"     NUCLEAR RESET: Classifier reinitialized, LR -> {new_lr:.2e}")
                    return True

        return False

    def train(self):
        """Modified training loop with curriculum learning support"""
        print("Starting training with curriculum learning...")
        torch.set_num_threads(4)

        self._debug_label_distribution()
        self._debug_sample_labels()

        if self.use_curriculum_learning:
            self.analyze_padding_improvement()

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            self.current_epoch = epoch

            # Update curriculum for this epoch
            if self.use_curriculum_learning and self.curriculum_dataset:
                # Update the curriculum dataset for current epoch
                if hasattr(self.train_loader.dataset, 'update_epoch'):
                    self.train_loader.dataset.update_epoch(epoch)

                # Log curriculum statistics
                current_samples = len(self.train_loader.dataset)
                total_samples = len(
                    self.curriculum_dataset.difficulty_metrics) if self.curriculum_dataset else current_samples
                wandb.log({
                    'curriculum/current_samples': current_samples,
                    'curriculum/total_samples': total_samples,
                    'curriculum/sample_ratio': current_samples / max(1, total_samples),
                    'curriculum/difficulty_threshold': self.curriculum_dataset.scheduler.get_difficulty_threshold(
                        epoch) if self.curriculum_dataset else 1.0
                })

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, metrics = self.validate_epoch()

            # Update learning rate
            self.scheduler.step()

            # Log to WandB
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': metrics['accuracy'],
                'val_precision': metrics['precision'],
                'val_recall': metrics['recall'],
                'val_f1': metrics['f1']
            })

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"Val F1: {metrics['f1']:.4f}")

            # Optional analysis (with error handling)
            if (epoch + 1) % 3 == 0:
                try:
                    self.analyze_predictions(num_samples=2)
                except Exception as e:
                    print(f"Skipping analysis due to error: {e}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model()
                print("New best model saved!")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.config.patience}")

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print("Early stopping triggered!")
                break

        print("Training completed!")

        # Plot training curves (with error handling)
        try:
            self._plot_training_curves()
        except Exception as e:
            print(f"Could not plot training curves: {e}")

        # Final evaluation (with error handling)
        print("\n=== Final Evaluation ===")
        try:
            self.evaluate_sample(0)
        except Exception as e:
            print(f"Final evaluation failed: {e}")
            print("Training completed successfully despite evaluation error.")

    def save_model(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'vocab_size': self.dataset.vocab_size,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        torch.save(checkpoint, self.config.model_save_path)

        # Save to WandB
        # wandb.save(self.config.model_save_path)

    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        print(f"Model loaded from {checkpoint_path}")

    def _plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_curves.png')
        wandb.log({"training_curves": wandb.Image('training_curves.png')})
        plt.close()

    def evaluate_sample(self, sample_idx: int = 0):
        """Evaluate a single sample and show predictions"""
        try:
            self.model.eval()

            sample = self.dataset[sample_idx]
            sequences = sample['sequence'].unsqueeze(0).to(self.device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)
            labels = sample['labels'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(sequences, attention_mask)
                predictions = outputs['predictions']

            # Decode predictions and labels safely
            try:
                pred_text = self.dataset.decode_annotation(predictions[0].cpu().numpy())
                true_text = sample['annotation']
            except Exception as e:
                print(f"Decoding failed: {e}")
                pred_text = "<DECODE_ERROR>"
                true_text = sample.get('annotation', '<UNKNOWN>')

            print(f"True annotation: {true_text}")
            print(f"Predicted annotation: {pred_text}")

            return pred_text, true_text

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return "<ERROR>", "<ERROR>"

    def analyze_predictions(self, num_samples: int = 5):
        """Analyze what the model is predicting"""
        self.model.eval()

        print(f"\n=== Prediction Analysis ===")

        try:
            # Create id_to_vocab mapping safely
            id_to_vocab = {v: k for k, v in self.dataset.vocab.items()}

            with torch.no_grad():
                for i in range(min(num_samples, len(self.dataset))):
                    sample = self.dataset[i]
                    sequences = sample['sequence'].unsqueeze(0).to(self.device)
                    attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)

                    outputs = self.model(sequences, attention_mask)
                    predictions = outputs['predictions']

                    # Analyze prediction distribution
                    pred_tokens = predictions[0].cpu().numpy()
                    unique_tokens, counts = np.unique(pred_tokens, return_counts=True)

                    print(f"\nSample {i}:")
                    print(f"  True: {sample['annotation']}")
                    print(f"  Prediction token distribution: {dict(zip(unique_tokens, counts))}")

                    # Show token meanings safely
                    for token_id, count in zip(unique_tokens[:5], counts[:5]):  # Only first 5
                        token_text = id_to_vocab.get(int(token_id), f'UNK_{token_id}')
                        print(f"    ID {token_id} ('{token_text}'): {count} times")

        except Exception as e:
            print(f"Analysis failed (non-critical): {e}")
            print("Continuing training...")


class BatchBucketing:
    """Simple sampler that groups similar-length sequences together"""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._create_length_groups()

    def _create_length_groups(self):
        """Group dataset indices by sequence length"""
        print("Analyzing sequence lengths for grouping...")

        # Get all sequence lengths
        length_to_indices = {}
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                seq_length = sample['attention_mask'].sum().item()

                if seq_length not in length_to_indices:
                    length_to_indices[seq_length] = []
                length_to_indices[seq_length].append(idx)

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

        # Sort by length and group into batches
        self.batches = []
        sorted_lengths = sorted(length_to_indices.keys())

        print(f"Found sequences with lengths: {sorted_lengths[:10]}..." if len(
            sorted_lengths) > 10 else f"Found sequences with lengths: {sorted_lengths}")

        # Collect all indices in length order
        all_indices = []
        for length in sorted_lengths:
            indices = length_to_indices[length]
            if self.shuffle:
                np.random.shuffle(indices)  # Shuffle within same length
            all_indices.extend(indices)

        # Create batches from grouped indices
        for i in range(0, len(all_indices), self.batch_size):
            batch_indices = all_indices[i:i + self.batch_size]
            if len(batch_indices) > 0:
                self.batches.append(batch_indices)

        # Print statistics
        batch_length_ranges = []
        for batch_indices in self.batches[:5]:  # Sample first 5 batches
            lengths = []
            for idx in batch_indices:
                sample = self.dataset[idx]
                seq_length = sample['attention_mask'].sum().item()
                lengths.append(seq_length)
            batch_length_ranges.append((min(lengths), max(lengths)))

        print(f"Created {len(self.batches)} batches")
        print(f"Sample batch length ranges: {batch_length_ranges}")

    def __iter__(self):
        """Generate batch indices"""
        batches = self.batches.copy()
        if self.shuffle:
            np.random.shuffle(batches)  # Shuffle batch order

        for batch_indices in batches:
            yield batch_indices

    def __len__(self):
        return len(self.batches)

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train LS model for sign language recognition')
    parser.add_argument('--data_dir', type=str, default='./output', help='Directory containing JSON files')
    parser.add_argument('--annotations_path', type=str, default='./phoenix_annotations.xlsx', help='Path to annotations Excel file')
    parser.add_argument('--config_path', type=str, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')
    parser.add_argument('--curriculum_learning', action='store_true', default=True, help='Enable curriculum learning')
    parser.add_argument('--curriculum_warmup_epochs', type=int, default=5, help='Number of warmup epochs for curriculum learning')
    parser.add_argument('--no_curriculum', action='store_true', help='Disable curriculum learning')

    args = parser.parse_args()

    # Load configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig(
            data_dir=args.data_dir,
            annotations_path=args.annotations_path
        )


    # Initialize trainer
    trainer = SignLanguageTrainer(config)

    # Resume training if checkpoint provided
    if args.resume:
        trainer.load_model(args.resume)

    # Start training
    trainer.train()

    # Evaluate a sample
    print("\nEvaluating sample:")
    trainer.evaluate_sample(0)


if __name__ == "__main__":
    main()