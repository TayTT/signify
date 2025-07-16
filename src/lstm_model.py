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
            bidirectional=config.bidirectional,  # False for real-time
            batch_first=True
        )

        # Attention mechanism
        lstm_output_size = config.hidden_size  # No *2 since not bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

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
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for training (batch_size, target_seq_len)

        Returns:
            Dictionary with logits, loss, and predictions
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
            # Convert attention mask to key padding mask
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None

        attn_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )

        # Residual connection and layer norm
        x = self.layer_norm(lstm_out + attn_out)
        x = self.dropout(x)

        # Classification
        logits = self.classifier(x)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.use_ctc:
                # CTC loss for sequence alignment
                log_probs = F.log_softmax(logits, dim=-1)
                input_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.full((batch_size,), seq_len)
                target_lengths = (labels != 0).sum(dim=1)  # Assuming 0 is padding token

                loss = F.ctc_loss(
                    log_probs.transpose(0, 1),  # (seq_len, batch_size, vocab_size)
                    labels,
                    input_lengths,
                    target_lengths,
                    blank=0,  # Assuming 0 is blank/padding token
                    reduction='mean'
                )
            else:
                # Standard cross-entropy loss
                loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        return {
            'logits': logits,
            'loss': loss,
            'attention_weights': attention_weights,
            'predictions': torch.argmax(logits, dim=-1)
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
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Training tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

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
        """Create train and validation data loaders"""
        # Split dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders with reduced num_workers and better error handling
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            pin_memory=False,  # Disable pin_memory since no GPU acceleration warning
            collate_fn=self._safe_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            pin_memory=False,  # Disable pin_memory since no GPU acceleration warning
            collate_fn=self._safe_collate_fn
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    def _safe_collate_fn(self, batch):
        """Safe collate function that handles shape mismatches"""
        try:
            # Check if all samples have the same shapes
            sequences = [item['sequence'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            labels = [item['labels'] for item in batch]

            # Verify shapes
            seq_shapes = [seq.shape for seq in sequences]
            mask_shapes = [mask.shape for mask in attention_masks]
            label_shapes = [label.shape for label in labels]

            if len(set(seq_shapes)) > 1:
                print(f"Warning: Inconsistent sequence shapes in batch: {seq_shapes}")
                # Fix by using the most common shape or a default shape
                target_shape = seq_shapes[0]  # Use first item's shape as reference
                fixed_sequences = []
                for seq in sequences:
                    if seq.shape != target_shape:
                        fixed_seq = torch.zeros(target_shape, dtype=seq.dtype)
                        # Copy as much as possible
                        min_dims = [min(seq.shape[i], target_shape[i]) for i in range(len(target_shape))]
                        if len(min_dims) == 2:
                            fixed_seq[:min_dims[0], :min_dims[1]] = seq[:min_dims[0], :min_dims[1]]
                        else:
                            fixed_seq[:min_dims[0]] = seq[:min_dims[0]]
                        fixed_sequences.append(fixed_seq)
                    else:
                        fixed_sequences.append(seq)
                sequences = fixed_sequences

            # Stack tensors
            return {
                'sequence': torch.stack(sequences),
                'attention_mask': torch.stack(attention_masks),
                'labels': torch.stack(labels),
                'annotation': [item['annotation'] for item in batch],
                'metadata': [item['metadata'] for item in batch]
            }
        except Exception as e:
            print(f"Error in collate function: {e}")
            # Return a minimal valid batch
            batch_size = len(batch)
            return {
                'sequence': torch.zeros(batch_size, self.dataset.preprocessor.config.max_sequence_length,
                                        self.dataset.preprocessor.feature_dims['total']),
                'attention_mask': torch.zeros(batch_size, self.dataset.preprocessor.config.max_sequence_length,
                                              dtype=torch.bool),
                'labels': torch.zeros(batch_size, 50, dtype=torch.long),
                'annotation': [''] * batch_size,
                'metadata': [{}] * batch_size
            }

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            sequences = batch['sequence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequences, attention_mask, labels)
            loss = outputs['loss']

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update parameters
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to WandB
            wandb.log({
                'train_loss_step': loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

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

    def train(self):
        """Main training loop"""
        print("Starting training...")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, metrics = self.validate_epoch()

            # Update learning rate
            self.scheduler.step(val_loss)

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

        # Plot training curves
        self._plot_training_curves()

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
        wandb.save(self.config.model_save_path)

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
        self.model.eval()

        sample = self.dataset[sample_idx]
        sequences = sample['sequence'].unsqueeze(0).to(self.device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)
        labels = sample['labels'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequences, attention_mask)
            predictions = outputs['predictions']

        # Decode predictions and labels
        pred_text = self.dataset.decode_annotation(predictions[0].cpu().numpy())
        true_text = sample['annotation']

        print(f"True annotation: {true_text}")
        print(f"Predicted annotation: {pred_text}")

        return pred_text, true_text


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