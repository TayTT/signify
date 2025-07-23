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

from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig, PhoenixDataset


@dataclass
@dataclass
@dataclass
class ModelConfig:
    """Configuration for LSTM model"""
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
        self.use_ctc = True

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
            else:
                # FOCAL LOSS: Focus on hard examples to prevent easy convergence
                logits_flat = logits[:, :labels.shape[1], :].reshape(-1, self.vocab_size)
                labels_flat = labels.reshape(-1)

                # Calculate cross-entropy without reduction
                ce_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0, reduction='none')

                # Apply focal loss weighting
                pt = torch.exp(-ce_loss)  # Probability of true class
                focal_weight = (1 - pt) ** 2  # Focus on hard examples (low pt)
                focal_loss = focal_weight * ce_loss

                # Only average over non-padding tokens
                mask = labels_flat != 0
                if mask.sum() > 0:
                    loss = focal_loss[mask].mean()
                else:
                    loss = focal_loss.mean()  # Fallback if all padding

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

        return train_loader, val_loader

    # def _safe_collate_fn(self, batch):
    #     """Safe collate function that handles shape mismatches"""
    #     try:
    #         # Check if all samples have the same shapes
    #         sequences = [item['sequence'] for item in batch]
    #         attention_masks = [item['attention_mask'] for item in batch]
    #         labels = [item['labels'] for item in batch]
    #
    #         # Verify shapes
    #         seq_shapes = [seq.shape for seq in sequences]
    #         mask_shapes = [mask.shape for mask in attention_masks]
    #         label_shapes = [label.shape for label in labels]
    #
    #         if len(set(seq_shapes)) > 1:
    #             print(f"Warning: Inconsistent sequence shapes in batch: {seq_shapes}")
    #             # Fix by using the most common shape or a default shape
    #             target_shape = seq_shapes[0]  # Use first item's shape as reference
    #             fixed_sequences = []
    #             for seq in sequences:
    #                 if seq.shape != target_shape:
    #                     fixed_seq = torch.zeros(target_shape, dtype=seq.dtype)
    #                     # Copy as much as possible
    #                     min_dims = [min(seq.shape[i], target_shape[i]) for i in range(len(target_shape))]
    #                     if len(min_dims) == 2:
    #                         fixed_seq[:min_dims[0], :min_dims[1]] = seq[:min_dims[0], :min_dims[1]]
    #                     else:
    #                         fixed_seq[:min_dims[0]] = seq[:min_dims[0]]
    #                     fixed_sequences.append(fixed_seq)
    #                 else:
    #                     fixed_sequences.append(seq)
    #             sequences = fixed_sequences
    #
    #         # Stack tensors
    #         return {
    #             'sequence': torch.stack(sequences),
    #             'attention_mask': torch.stack(attention_masks),
    #             'labels': torch.stack(labels),
    #             'annotation': [item['annotation'] for item in batch],
    #             'metadata': [item['metadata'] for item in batch]
    #         }
    #     except Exception as e:
    #         print(f"Error in collate function: {e}")
    #         # Return a minimal valid batch
    #         batch_size = len(batch)
    #         return {
    #             'sequence': torch.zeros(batch_size, self.dataset.preprocessor.config.max_sequence_length,
    #                                     self.dataset.preprocessor.feature_dims['total']),
    #             'attention_mask': torch.zeros(batch_size, self.dataset.preprocessor.config.max_sequence_length,
    #                                           dtype=torch.bool),
    #             'labels': torch.zeros(batch_size, 50, dtype=torch.long),
    #             'annotation': [''] * batch_size,
    #             'metadata': [{}] * batch_size
    #         }

    def _dynamic_collate_fn(self, batch):
        """Dynamic collate function that pads only to batch maximum length"""
        try:
            # Extract all components
            sequences = [item['sequence'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            labels = [item['labels'] for item in batch]
            annotations = [item['annotation'] for item in batch]
            metadata = [item['metadata'] for item in batch]

            # Find actual lengths in this batch
            actual_lengths = [mask.sum().item() for mask in attention_masks]
            batch_max_seq_length = max(actual_lengths)
            batch_min_seq_length = min(actual_lengths)

            # Don't make batches smaller than a minimum size (for stability)
            min_batch_length = 32
            batch_max_seq_length = max(batch_max_seq_length, min_batch_length)

            # Calculate padding efficiency
            total_original_padding = len(sequences) * self.config.max_sequence_length - sum(actual_lengths)
            total_new_padding = len(sequences) * batch_max_seq_length - sum(actual_lengths)
            padding_reduction = total_original_padding - total_new_padding

            # Occasionally print stats (every 50 batches)
            if np.random.random() < 0.02:  # ~2% of batches
                print(f"Batch stats: lengths {batch_min_seq_length}-{batch_max_seq_length}, "
                      f"padding_to={batch_max_seq_length}, "
                      f"saved_padding={padding_reduction} tokens ({padding_reduction / (total_original_padding or 1) * 100:.1f}%)")

            # Dynamically resize sequences to batch maximum
            resized_sequences = []
            resized_attention_masks = []

            for seq, mask in zip(sequences, attention_masks):
                # Truncate or pad sequence to batch_max_seq_length
                if seq.shape[0] >= batch_max_seq_length:
                    # Truncate
                    resized_seq = seq[:batch_max_seq_length]
                    resized_mask = mask[:batch_max_seq_length]
                else:
                    # Pad to batch max
                    pad_length = batch_max_seq_length - seq.shape[0]
                    padding = torch.zeros(pad_length, seq.shape[1], dtype=seq.dtype)
                    mask_padding = torch.zeros(pad_length, dtype=mask.dtype)

                    resized_seq = torch.cat([seq, padding], dim=0)
                    resized_mask = torch.cat([mask, mask_padding], dim=0)

                resized_sequences.append(resized_seq)
                resized_attention_masks.append(resized_mask)

            # Handle labels (keep original max length for labels)
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

            # Stack everything
            return {
                'sequence': torch.stack(resized_sequences),
                'attention_mask': torch.stack(resized_attention_masks),
                'labels': torch.stack(resized_labels),
                'annotation': annotations,
                'metadata': metadata
            }

        except Exception as e:
            print(f"Error in dynamic collate function: {e}")
            # Fallback to original behavior
            return self._safe_collate_fn(batch)

    def train_epoch(self) -> float:
        """Train for one epoch with padding ratio monitoring"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        # Padding ratio tracking
        padding_ratios = []
        sample_frequency = max(1, num_batches // 10)  # Sample ~10 batches per epoch

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            sequences = batch['sequence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Calculate padding ratio for this batch
            total_tokens = sequences.numel()  # Total elements in sequences tensor
            actual_tokens = attention_mask.sum().item()  # Non-padding tokens
            padding_tokens = total_tokens - actual_tokens
            padding_ratio = (padding_tokens / total_tokens) * 100 if total_tokens > 0 else 0

            padding_ratios.append(padding_ratio)

            # Display sample padding ratios during training
            if batch_idx % sample_frequency == 0 or batch_idx < 3:  # Show first 3 + samples
                batch_size, seq_len, features = sequences.shape
                avg_seq_length = actual_tokens / batch_size
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

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequences, attention_mask, labels)
            loss = outputs['loss']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

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
                current_total_tokens = sequences.numel()
                current_actual_tokens = attention_mask.sum().item()
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

    def train(self):
        """Main training loop"""
        print("Starting training...")
        torch.set_num_threads(4)

        self.analyze_padding_improvement()

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

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
            if (epoch + 1) % 3 == 0:  # Every 3 epochs, less frequent
                try:
                    self.analyze_predictions(num_samples=2)  # Fewer samples
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
    parser = argparse.ArgumentParser(description='Train LSTM model for sign language recognition')
    parser.add_argument('--data_dir', type=str, default='./output', help='Directory containing JSON files')
    parser.add_argument('--annotations_path', type=str, default='./phoenix_annotations.xlsx', help='Path to annotations Excel file')
    parser.add_argument('--config_path', type=str, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')

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