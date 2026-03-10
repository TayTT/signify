"""
LSTM Model for Sign Language Recognition (Sign2Gloss)

This module implements a Bi-LSTM model for converting sign language video sequences
to gloss annotations, ready for further processing with mBART.

Configuration is now handled through the config module. Use load_config() to load
settings from a YAML/JSON file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import argparse
from dataclasses import dataclass, asdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Import configuration system
from config import Config, ModelConfig, load_config

from preprocess_jsons import SignLanguagePreprocessor, PhoenixDataset


DEBUG = True


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

    def __init__(self, config: Config, vocab_size: int):
        super().__init__()

        self.config = config
        model_cfg = config.model
        self.vocab_size = vocab_size

        # Extract model parameters
        input_size = model_cfg.input_size
        hidden_size = model_cfg.hidden_size
        num_layers = model_cfg.num_layers
        dropout = model_cfg.dropout
        bidirectional = model_cfg.bidirectional
        max_sequence_length = config.preprocessing.max_sequence_length  # owned by preprocessing

        print(f"Initializing LSTM model:")
        print(f"  - Input size: {input_size}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Vocab size: {vocab_size}")

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_sequence_length)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_size = hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, vocab_size)
        )

        self.use_ctc = False

    def forward(self, x, attention_mask=None, labels=None):
        """Forward pass"""
        batch_size, seq_len, _ = x.shape

        x = self.input_projection(x)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        lstm_out, (hidden, cell) = self.lstm(x)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None

        attn_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )

        x = self.layer_norm(lstm_out + attn_out)
        x = self.dropout(x)

        logits = self.classifier(x)
        predictions = torch.argmax(logits, dim=-1)

        loss = None
        if labels is not None:
            if self.use_ctc:
                log_probs = F.log_softmax(logits, dim=-1)
                input_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.full((batch_size,), seq_len)
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
                logits_flat = logits[:, :labels.shape[1], :].reshape(-1, self.vocab_size)
                labels_flat = labels.reshape(-1)

                class_weights = torch.ones(self.vocab_size, device=logits.device)
                class_weights[0] = 0.1

                penalty_weights = torch.ones_like(labels_flat, device=logits.device, dtype=torch.float)

                for batch_idx in range(labels.shape[0]):
                    batch_labels = labels[batch_idx]
                    non_padding_mask = batch_labels != 0
                    if non_padding_mask.sum() > 0:
                        natural_length = non_padding_mask.sum().item()
                        batch_start = batch_idx * labels.shape[1]
                        early_eos_threshold = int(natural_length * 0.5)

                        for pos in range(1, min(early_eos_threshold, labels.shape[1])):
                            global_pos = batch_start + pos
                            if global_pos < len(labels_flat):
                                if labels_flat[global_pos] == 3:
                                    penalty_weights[global_pos] = 5.0
                                elif labels_flat[global_pos] == 2:
                                    penalty_weights[global_pos] = 6.0

                ce_loss = F.cross_entropy(logits_flat, labels_flat, weight=class_weights, ignore_index=0, reduction='none')
                weighted_loss = ce_loss * penalty_weights

                probs = F.softmax(logits_flat, dim=-1)
                padding_prob = probs[:, 0]
                diversity_penalty = torch.mean(padding_prob) * 1.0

                mask = labels_flat != 0
                if mask.sum() > 0:
                    loss = weighted_loss[mask].mean() + diversity_penalty
                else:
                    loss = weighted_loss.mean() + diversity_penalty

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


class SignLanguageTrainer:
    """Training pipeline for sign language LSTM model"""

    def __init__(self, config: Config, feature_config: dict = None):
        self.config = config
        self.device = torch.device(config.device)
        self.use_curriculum_learning = False
        self.curriculum_dataset = None
        self.current_epoch = 0

        wandb.init(
            project=self.config.logging.project_name,
            name=self.config.logging.experiment_name,
            config=self.config.to_dict()
        )

        Path(self.config.data.model_save_path).parent.mkdir(parents=True, exist_ok=True)

        self.preprocessor = SignLanguagePreprocessor(self.config.preprocessing)

        self.dataset = self._load_dataset()

        # update input size based on actual feature dimensions
        actual_input_size = self.preprocessor.feature_dims['total']
        if self.config.model.input_size != actual_input_size:
            print(f"Updating input_size from {self.config.model.input_size} to {actual_input_size}")
            self.config.model.input_size = actual_input_size

        self.train_loader, self.val_loader = self._create_data_loaders()

        self.model = SignLanguageLSTM(self.config, self.dataset.vocab_size).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.optimizer.learning_rate,
            weight_decay=self.config.training.optimizer.weight_decay,
            betas=(self.config.training.optimizer.beta1, self.config.training.optimizer.beta2),
            eps=self.config.training.optimizer.eps
        )

        warmup_steps = self.config.training.scheduler.warmup_steps
        total_steps = self.config.training.scheduler.total_steps

        def lr_lambda_func(step):
            if step < warmup_steps:
                return max(0.1, step / warmup_steps)
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                progress = max(0.0, min(1.0, progress))
                import math
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda_func)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def _load_dataset(self) -> PhoenixDataset:
        """Load and create Phoenix dataset"""
        try:
            vocab_path = Path(self.config.data.vocab_path)
            dataset = self.preprocessor.create_phoenix_dataset(
                data_dir=self.config.data.data_dir,
                annotations_path=self.config.data.annotations_path,
                vocab_path=str(vocab_path) if vocab_path.exists() else None
            )

            dataset.save_vocabulary(self.config.data.vocab_path)

            print(f"Loaded dataset with {len(dataset)} samples")
            print(f"Vocabulary size: {dataset.vocab_size}")

            return dataset

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders"""
        train_size = int(self.config.training.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.training.random_seed)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._dynamic_collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._dynamic_collate_fn,
            drop_last=False
        )

        print(f"Train samples: {len(train_dataset)} -> ~{len(train_loader)} batches")
        print(f"Validation samples: {len(val_dataset)} -> ~{len(val_loader)} batches")

        return train_loader, val_loader

    def _dynamic_collate_fn(self, batch):
        """Dynamic collate function with curriculum-aware sequence lengths"""
        try:
            sequences = [item['sequence'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            labels = [item['labels'] for item in batch]
            annotations = [item['annotation'] for item in batch]
            metadata = [item['metadata'] for item in batch]

            if len(set(seq.shape[0] for seq in sequences)) > 1:
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
                resized_sequences = sequences
                resized_attention_masks = attention_masks

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

            return {
                'sequence': torch.stack(resized_sequences),
                'attention_mask': torch.stack(resized_attention_masks),
                'labels': torch.stack(resized_labels),
                'annotation': annotations,
                'metadata': metadata
            }

        except Exception as e:
            print(f"Error in dynamic collate function: {e}")
            raise

    def _calculate_padding_ratio(self, sequences: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """Calculate padding ratio"""
        total_elements = sequences.numel()
        non_padding_positions = attention_mask.sum().item()
        feature_dims = sequences.shape[2]
        actual_elements = non_padding_positions * feature_dims
        padding_elements = total_elements - actual_elements
        padding_ratio = (padding_elements / total_elements) * 100 if total_elements > 0 else 0
        return padding_ratio

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        padding_ratios = []

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            sequences = batch['sequence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            if DEBUG and batch_idx == 0:
                print(f"seq NaN: {torch.isnan(sequences).any()}, inf: {torch.isinf(sequences).any()}")
                print(f"mask NaN: {torch.isnan(attention_mask.float()).any()}")
                print(f"labels NaN: {torch.isnan(labels.float()).any()}")
                print(f"seq range: [{sequences.min():.3f}, {sequences.max():.3f}]")
                print(f"attention_mask all zeros: {(attention_mask.sum(dim=1) == 0).any()}")

            padding_ratio = self._calculate_padding_ratio(sequences, attention_mask)
            padding_ratios.append(padding_ratio)

            self.optimizer.zero_grad()

            outputs = self.model(sequences, attention_mask, labels)
            loss = outputs['loss']

            if DEBUG and batch_idx == 0:
                print(f"logits NaN: {torch.isnan(outputs['logits']).any()}")
                print(f"loss value: {loss.item()}")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN loss at batch {batch_idx}!")
                print(f"seq NaN: {torch.isnan(sequences).any()}, range: [{sequences.min():.3f}, {sequences.max():.3f}]")
                print(f"logits NaN: {torch.isnan(outputs['logits']).any()}")
                print(f"attention_mask all zeros: {(attention_mask.sum(dim=1) == 0).any()}")
                self.optimizer.zero_grad()
                continue

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=self.config.training.gradient_clip_norm)
            if DEBUG and batch_idx < 3:
                print(f"batch {batch_idx} grad norm (pre-clip): {grad_norm:.4f}")

            self.optimizer.step()

            total_loss += loss.item()

            current_padding = padding_ratios[-1] if padding_ratios else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pad%': f'{current_padding:.1f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            wandb.log({
                'train_loss_step': loss.item(),
                'padding_ratio_batch': padding_ratio,
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
                sequences = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(sequences, attention_mask, labels)
                loss = outputs['loss']

                total_loss += loss.item()

                predictions = outputs['predictions'].cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels_np)

                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        metrics = self._calculate_metrics(all_predictions, all_labels)

        return avg_loss, metrics

    def _calculate_metrics(self, predictions: List, labels: List) -> Dict:
        """Calculate evaluation metrics"""
        pred_flat = []
        label_flat = []

        for pred, label in zip(predictions, labels):
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            if hasattr(label, 'cpu'):
                label = label.cpu().numpy()

            if np.isscalar(pred) and np.isscalar(label):
                if label != 0:
                    pred_flat.append(int(pred))
                    label_flat.append(int(label))
            else:
                pred = np.atleast_1d(pred)
                label = np.atleast_1d(label)
                min_len = min(len(pred), len(label))
                pred_trimmed = pred[:min_len]
                label_trimmed = label[:min_len]
                mask = label_trimmed != 0

                if mask.any():
                    pred_flat.extend(pred_trimmed[mask].tolist())
                    label_flat.extend(label_trimmed[mask].tolist())

        if not pred_flat or not label_flat:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'total_samples': 0,
                'prediction_diversity': 0.0
            }

        try:
            pred_flat = np.array(pred_flat)
            label_flat = np.array(label_flat)

            accuracy = accuracy_score(label_flat, pred_flat)
            precision, recall, f1, _ = precision_recall_fscore_support(
                label_flat, pred_flat, average='weighted', zero_division=0
            )

            total_samples = len(pred_flat)
            most_common_pred = np.bincount(pred_flat).max()
            prediction_diversity = 1.0 - (most_common_pred / total_samples)

            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'total_samples': total_samples,
                'prediction_diversity': float(prediction_diversity)
            }

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'total_samples': 0,
                'prediction_diversity': 0.0
            }

    def train(self):
        """Main training loop"""
        print("Starting training...")
        torch.set_num_threads(4)

        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_loss, metrics = self.validate_epoch()

            self.scheduler.step()

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': metrics['accuracy'],
                'val_precision': metrics['precision'],
                'val_recall': metrics['recall'],
                'val_f1': metrics['f1']
            })

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"Val F1: {metrics['f1']:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model()
                print("New best model saved!")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.config.training.patience}")

            if self.patience_counter >= self.config.training.patience:
                print("Early stopping triggered!")
                break

        print("Training completed!")

    def save_model(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'vocab_size': self.dataset.vocab_size,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'gloss_to_idx': self.dataset.vocab,
            'idx_to_gloss': {v: k for k, v in self.dataset.vocab.items()}
        }
        torch.save(checkpoint, self.config.data.model_save_path)

    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        # Note: config is now a dictionary, not an object
        # If you need to use it, convert back to Config object:
        # from config import Config
        # loaded_config = Config.from_dict(checkpoint['config'])

        print(f"Model loaded from {checkpoint_path}")

    def evaluate_sample(self, sample_idx: int) -> Tuple[str, str]:
        """Evaluate a single sample"""
        self.model.eval()

        try:
            sample = self.dataset[sample_idx]
            sequences = sample['sequence'].unsqueeze(0).to(self.device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(sequences, attention_mask)
                predictions = outputs['predictions'][0].cpu().numpy()

            id_to_vocab = {v: k for k, v in self.dataset.vocab.items()}

            pred_tokens = []
            for token_id in predictions:
                if token_id > 3:
                    token = id_to_vocab.get(int(token_id), f'UNK_{token_id}')
                    if not pred_tokens or pred_tokens[-1] != token:
                        pred_tokens.append(token)

            pred_text = ' '.join(pred_tokens)
            true_text = sample['annotation']

            print(f"True: {true_text}")
            print(f"Pred: {pred_text}")

            return pred_text, true_text

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return "<e>", "<e>"


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train LSTM model for sign language recognition')
    parser.add_argument('--config', type=str, help='Path to config YAML/JSON file')
    parser.add_argument('--data_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('--annotations_path', type=str, help='Path to annotations Excel file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')
    parser.add_argument('--no_faces', action='store_true', help='Exclude face landmarks')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)

        # Apply command-line overrides
        if args.data_dir:
            config.data.data_dir = args.data_dir
        if args.annotations_path:
            config.data.annotations_path = args.annotations_path
    else:
        # Create config from command-line arguments
        config = Config()
        if args.data_dir:
            config.data.data_dir = args.data_dir
        if args.annotations_path:
            config.data.annotations_path = args.annotations_path

    feature_config = {
        'include_face': not args.no_faces
    }

    trainer = SignLanguageTrainer(config, feature_config)

    if args.resume:
        trainer.load_model(args.resume)

    trainer.train()

    print("\nEvaluating sample:")
    trainer.evaluate_sample(0)


if __name__ == "__main__":
    main()