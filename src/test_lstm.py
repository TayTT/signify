#!/usr/bin/env python3
"""
Test and Evaluate Trained LSTM Sign Language Model

This script provides multiple ways to test your trained model:
1. Evaluate on validation/test set
2. Test on individual videos
3. Generate detailed metrics and confusion matrix
4. Export predictions to file

Usage:
    # Test on validation set
    python test_model.py \
        --model_path ./loss14/lstm_sign2gloss.pth \
        --data_dir ./data/phoenix_test \
        --annotations_path ./data/test_corpus.csv

    # Test single video
    python test_model.py \
        --model_path ./loss14/lstm_sign2gloss.pth \
        --video_path ./video.mp4 \
        --landmarks_path ./video_landmarks.json
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, load_config
from lstm_model import SignLanguageLSTM, SignLanguageTrainer
from preprocess_jsons import SignLanguagePreprocessor, PhoenixDataset
from train_lstm import PhoenixDatasetManager


class ModelTester:
    """Comprehensive model testing and evaluation"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"Loading model from: {self.model_path}")
        self.load_model()

    def load_model(self):
        """Load trained model from checkpoint"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        config_data = checkpoint.get('config')

        if isinstance(config_data, Config):
            self.config = config_data
        else:
            # old flat checkpoint - reconstruct a Config from whatever attrs exist
            print("Old checkpoint format, rebuilding config...")
            self.config = Config()
            self.config.model.input_size = getattr(config_data, 'input_size', 345)
            self.config.model.hidden_size = getattr(config_data, 'hidden_size', 768)
            self.config.model.num_layers = getattr(config_data, 'num_layers', 1)
            self.config.model.dropout = getattr(config_data, 'dropout', 0.1)
            self.config.model.bidirectional = getattr(config_data, 'bidirectional', False)
            self.config.model.max_annotation_length = getattr(config_data, 'max_annotation_length', 25)
            self.config.preprocessing.max_sequence_length = getattr(config_data, 'max_sequence_length', 224)
            self.config.training.batch_size = getattr(config_data, 'batch_size', 4)
            self.config.device = str(self.device)

        self.vocab_size = checkpoint['model_state_dict']['classifier.3.weight'].shape[0]

        self.gloss_to_idx = None
        self.idx_to_gloss = None

        if 'gloss_to_idx' in checkpoint:
            print("Loading vocabulary from checkpoint...")
            self.gloss_to_idx = checkpoint['gloss_to_idx']
            self.idx_to_gloss = checkpoint['idx_to_gloss']

        if self.gloss_to_idx is None:
            vocab_path = self.model_path.parent / "vocab.pkl"
            if vocab_path.exists():
                import pickle
                print(f"Loading vocabulary from {vocab_path}...")
                with open(vocab_path, 'rb') as f:
                    vocab_data = pickle.load(f)

                if isinstance(vocab_data, dict):
                    if 'gloss_to_idx' in vocab_data:
                        self.gloss_to_idx = vocab_data['gloss_to_idx']
                        self.idx_to_gloss = vocab_data['idx_to_gloss']
                    elif 'vocab' in vocab_data:
                        self.gloss_to_idx = vocab_data['vocab']
                        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
                    else:
                        self.gloss_to_idx = vocab_data
                        self.idx_to_gloss = {v: k for k, v in vocab_data.items()}

        if self.gloss_to_idx is None:
            print(f"Warning: Vocabulary not found - predictions will show token IDs")

        self.model = SignLanguageLSTM(self.config, self.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully")
        print(f"  Architecture: {self.config.model.num_layers} layers, {self.config.model.hidden_size} hidden units")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"  Device: {self.device}")

    def evaluate_on_dataset(self, data_dir: str, annotations_path: str,
                            split: str = 'test') -> Dict:
        """Evaluate model on a dataset"""
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {split} set")
        print(f"{'=' * 60}")

        # Load dataset
        dataset_manager = PhoenixDatasetManager(
            data_dir=data_dir,
            annotations_path=annotations_path
        )

        preprocessor = SignLanguagePreprocessor(self.config.preprocessing)  # use training config as-is

        print("Loading dataset...")
        dataset = dataset_manager.create_dataset(preprocessor)

        # Apply vocabulary from trained model if available
        if self.gloss_to_idx is not None:
            dataset.gloss_to_idx = self.gloss_to_idx
            dataset.idx_to_gloss = self.idx_to_gloss
            dataset.vocab_size = self.vocab_size
            print("Applied vocabulary from trained model")
        else:
            print("Warning: Using dataset's own vocabulary (may cause issues)")

        print(f"Dataset loaded: {len(dataset)} samples")

        # Custom collate function to handle variable-length sequences
        def collate_fn(batch):
            """Custom collate function to handle variable-length labels"""
            # Get max lengths in this batch
            max_seq_len = max(item['sequence'].shape[0] for item in batch)
            max_label_len = max(item['labels'].shape[0] for item in batch)

            # Prepare batched tensors
            sequences = []
            attention_masks = []
            labels = []

            for item in batch:
                seq = item['sequence']
                mask = item['attention_mask']
                label = item['labels']

                # Pad sequence if needed
                if seq.shape[0] < max_seq_len:
                    padding = torch.zeros(max_seq_len - seq.shape[0], seq.shape[1])
                    seq = torch.cat([seq, padding], dim=0)
                    mask_padding = torch.zeros(max_seq_len - mask.shape[0])
                    mask = torch.cat([mask, mask_padding], dim=0)

                # Pad labels if needed
                if label.shape[0] < max_label_len:
                    label_padding = torch.zeros(max_label_len - label.shape[0], dtype=label.dtype)
                    label = torch.cat([label, label_padding], dim=0)

                sequences.append(seq)
                attention_masks.append(mask)
                labels.append(label)

            return {
                'sequence': torch.stack(sequences),
                'attention_mask': torch.stack(attention_masks),
                'labels': torch.stack(labels),
                'annotation': [item['annotation'] for item in batch]
            }

        # Create data loader with custom collate
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn  # Add custom collate function
        )

        # Evaluate
        print("Running evaluation...")
        all_predictions = []
        all_labels = []
        all_losses = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                sequences = batch['sequence'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(sequences, attention_mask, labels)
                loss = outputs['loss']
                predictions = outputs['predictions']

                all_losses.append(loss.item())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels, dataset)
        metrics['avg_loss'] = np.mean(all_losses)

        # Print results
        self._print_results(metrics, split)

        return metrics

    def _calculate_metrics(self, predictions: List, labels: List,
                           dataset: PhoenixDataset) -> Dict:
        """Calculate evaluation metrics"""


        pred_flat = []
        label_flat = []

        for pred, label in zip(predictions, labels):
            pred = np.atleast_1d(pred)
            label = np.atleast_1d(label)

            min_len = min(len(pred), len(label))
            pred_trimmed = pred[:min_len]
            label_trimmed = label[:min_len]

            # Mask for non-padding tokens
            mask = label_trimmed != 0

            if mask.any():
                pred_flat.extend(pred_trimmed[mask])
                label_flat.extend(label_trimmed[mask])

        if not pred_flat or not label_flat:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'total_samples': len(predictions),
                'total_tokens': 0
            }

        # Calculate metrics
        accuracy = accuracy_score(label_flat, pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_flat, pred_flat, average='weighted', zero_division=0
        )

        # Calculate per-class metrics
        unique_labels = np.unique(label_flat)
        per_class_acc = {}

        for label_id in unique_labels[:20]:  # Top 20 classes
            mask = np.array(label_flat) == label_id
            if mask.sum() > 0:
                class_acc = np.mean(np.array(pred_flat)[mask] == label_id)
                gloss = dataset.idx_to_gloss.get(int(label_id), f"ID_{label_id}")
                per_class_acc[gloss] = class_acc

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_samples': len(predictions),
            'total_tokens': len(pred_flat),
            'per_class_accuracy': per_class_acc
        }

    def _print_results(self, metrics: Dict, split: str = 'test'):
        """Print evaluation results"""
        print(f"\n{'=' * 60}")
        print(f"{split.upper()} SET RESULTS")
        print(f"{'=' * 60}")

        print(f"\nOverall Metrics:")
        print(f"  Average Loss:      {metrics.get('avg_loss', 0):.4f}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1 Score:          {metrics['f1']:.4f}")
        print(f"  Total Samples:     {metrics['total_samples']}")
        print(f"  Total Tokens:      {metrics['total_tokens']}")

        if 'per_class_accuracy' in metrics and metrics['per_class_accuracy']:
            print(f"\nTop Classes by Accuracy:")
            sorted_classes = sorted(
                metrics['per_class_accuracy'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (gloss, acc) in enumerate(sorted_classes[:10]):
                print(f"  {i + 1:2d}. {gloss:20s}: {acc:.4f} ({acc * 100:.1f}%)")

        print(f"\n{'=' * 60}\n")

    def predict_single_sample(self, landmarks_path: str) -> str:
        """Predict gloss sequence for a single video"""
        print(f"Processing: {landmarks_path}")

        preprocessor = SignLanguagePreprocessor(self.config.preprocessing)
        result = preprocessor.process_video_file(landmarks_path)

        if result is None:
            raise ValueError(f"Failed to process video: {landmarks_path}")

        sequence = result['sequence'].unsqueeze(0).to(self.device)
        attention_mask = result['attention_mask'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequence, attention_mask)

        prediction_text = self._decode_output(outputs['logits'], attention_mask)
        print(f"Prediction: {prediction_text}")
        return prediction_text

    def _decode_output(self, logits, attention_mask):
        """CTC or argmax decode depending on model mode, returns gloss string"""
        if self.model.use_ctc:
            input_lengths = attention_mask.sum(dim=1).long()
            token_ids = SignLanguageLSTM.ctc_greedy_decode(logits, input_lengths)[0]
        else:
            preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            # collapse repeats, strip specials
            collapsed, prev = [], None
            for t in preds:
                if t != prev:
                    collapsed.append(t)
                prev = t
            token_ids = [t for t in collapsed if t > 3]

        if self.idx_to_gloss is None:
            return ' '.join(str(t) for t in token_ids)
        return ' '.join(self.idx_to_gloss.get(t, f'UNK_ID_{t}') for t in token_ids)

    def test_samples(self, dataset, num_samples: int = 10):
        """Test on random samples and show predictions"""
        print(f"\n{'=' * 60}")
        print(f"Testing on {num_samples} Random Samples")
        print(f"{'=' * 60}\n")

        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        correct = 0
        total = 0

        for i, idx in enumerate(indices):
            sample = dataset[idx]
            true_text = sample['annotation']

            sequence = sample['sequence'].unsqueeze(0).to(self.device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(sequence, attention_mask)

            pred_text = self._decode_output(outputs['logits'], attention_mask)

            is_correct = (pred_text.strip() == true_text.strip())
            correct += int(is_correct)
            total += 1

            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"Sample {i + 1}/{num_samples} [{status}]")
            print(f"  True: {true_text}")
            print(f"  Pred: {pred_text}")
            print()

        accuracy = correct / total if total > 0 else 0
        print(f"Sample Accuracy: {correct}/{total} ({accuracy * 100:.1f}%)\n")


def main():
    parser = argparse.ArgumentParser(description='Test trained LSTM model')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')

    # Testing mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test_dataset', action='store_true',
                       help='Evaluate on test dataset')
    group.add_argument('--test_video', type=str,
                       help='Test on single video (path to landmarks JSON/NPZ)')
    group.add_argument('--test_samples', type=int,
                       help='Test on N random samples from dataset')

    # Dataset arguments (for test_dataset mode)
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing test data')
    parser.add_argument('--annotations_path', type=str,
                        help='Path to test annotations')

    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str,
                        help='Path to save results (JSON)')

    args = parser.parse_args()

    # Initialize tester
    tester = ModelTester(args.model_path, args.device)

    # Run appropriate test
    if args.test_dataset:
        if not args.data_dir or not args.annotations_path:
            parser.error("--test_dataset requires --data_dir and --annotations_path")

        metrics = tester.evaluate_on_dataset(
            args.data_dir,
            args.annotations_path,
            split='test'
        )

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove non-serializable items
            save_metrics = {k: v for k, v in metrics.items()
                            if k != 'per_class_accuracy'}

            with open(output_path, 'w') as f:
                json.dump(save_metrics, f, indent=2)
            print(f"Results saved to: {output_path}")

    elif args.test_video:
        prediction = tester.predict_single_sample(args.test_video)

        if args.output:
            result = {'prediction': prediction, 'video_path': args.test_video}
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Prediction saved to: {args.output}")

    elif args.test_samples:
        if not args.data_dir or not args.annotations_path:
            parser.error("--test_samples requires --data_dir and --annotations_path")

        # Load dataset
        dataset_manager = PhoenixDatasetManager(
            data_dir=args.data_dir,
            annotations_path=args.annotations_path
        )

        preprocessor = SignLanguagePreprocessor(tester.config.preprocessing)
        dataset = dataset_manager.create_dataset(preprocessor)

        # Apply vocabulary
        if tester.gloss_to_idx is not None:
            dataset.vocab = tester.gloss_to_idx
            dataset.vocab_size = len(tester.gloss_to_idx)

        tester.test_samples(dataset, args.test_samples)


if __name__ == "__main__":
    main()