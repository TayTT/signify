#!/usr/bin/env python3
"""
Training Script for LSTM Sign Language Recognition Model

This script handles the complete training pipeline:
1. Process Phoenix dataset JSON files
2. Load annotations from Excel
3. Create datasets
4. Train LSTM model
5. Evaluate and save results

Usage:
    # Using config file:
    python train_lstm.py --config config.yaml

    # Using command-line arguments:
    python train_lstm.py --data_dir ./output --annotations_path ./annotations.csv

    # Resume training:
    python train_lstm.py --config config.yaml --resume ./loss14/checkpoint.pth
"""

import os
import sys
import argparse
import pandas as pd
import torch
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import asdict
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    Config, load_config, save_config, config_from_args,
    ModelConfig as LegacyModelConfig
)
from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig, PhoenixDataset
from lstm_model import SignLanguageLSTM, SignLanguageTrainer


class PhoenixDatasetManager:
    """Manages Phoenix dataset loading and processing"""

    def __init__(self, data_dir: str, annotations_path: str):
        self.data_dir = Path(data_dir)
        self.annotations_path = Path(annotations_path)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

    def load_annotations(self) -> pd.DataFrame:
        """Load annotations from Excel or CSV file"""
        try:
            print(f"Loading annotations from: {self.annotations_path}")

            df = None

            if self.annotations_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(self.annotations_path)
                print("Loaded Excel file")
            else:
                separators = ['|', ',', '\t', ';']

                print("Trying different separators...")
                for sep in separators:
                    try:
                        test_df = pd.read_csv(self.annotations_path, sep=sep, nrows=3)
                        print(f"  Separator '{sep}': {len(test_df.columns)} columns -> {list(test_df.columns)}")

                        if len(test_df.columns) >= 4:
                            df = pd.read_csv(self.annotations_path, sep=sep)
                            print(f"  Successfully loaded with separator '{sep}'")
                            break
                    except Exception as e:
                        print(f"  Separator '{sep}': Failed - {str(e)[:50]}...")
                        continue

            if df is None:
                raise ValueError("Failed to load annotations file")

            print(f"Initial shape: {df.shape}")
            print(f"Columns found: {list(df.columns)}")

            required_columns = ['id', 'folder', 'signer', 'annotation']

            if all(col in df.columns for col in required_columns):
                print("  All required columns found with exact names")
            else:
                print("Column names don't match exactly, attempting to map...")
                print(f"Expected: {required_columns}")
                print(f"Found: {list(df.columns)}")

                if len(df.columns) < 4:
                    raise ValueError(f"Insufficient columns: expected 4, found {len(df.columns)}")

                column_mapping = {}
                for i, req_col in enumerate(required_columns):
                    if i < len(df.columns):
                        old_col = df.columns[i]
                        column_mapping[old_col] = req_col

                print(f"Column mapping: {column_mapping}")
                df = df.rename(columns=column_mapping)

            df = df[required_columns].copy()

            initial_rows = len(df)
            df = df.dropna()
            df['id'] = df['id'].astype(str).str.strip()
            df['annotation'] = df['annotation'].astype(str).str.strip()
            df['signer'] = df['signer'].astype(str).str.strip()
            df['folder'] = df['folder'].astype(str).str.strip()

            df = df[df['annotation'].str.len() > 0]

            final_rows = len(df)
            if final_rows < initial_rows:
                print(f"  Removed {initial_rows - final_rows} rows with missing/empty data")

            print(f"  Final shape after cleaning: {df.shape}")
            print(f"  Data summary:")
            print(f"   - Unique signers: {df['signer'].nunique()}")
            print(f"   - Unique IDs: {df['id'].nunique()}")
            print(f"   - Sample annotations:")
            for i, annotation in enumerate(df['annotation'].head(3)):
                print(f"     {i + 1}. {annotation}")

            return df

        except Exception as e:
            print(f"  Error loading annotations: {e}")
            raise

    def find_landmark_files(self) -> Dict[str, str]:
        """Find all landmark files (NPZ or JSON) in the data directory"""
        landmark_files = {}

        npz_files = list(self.data_dir.rglob("*.npz"))
        if npz_files:
            print(f"Found {len(npz_files)} NPZ files (using NPZ format)")
            for npz_file in npz_files:
                identifier = npz_file.stem
                landmark_files[identifier] = str(npz_file)
        else:
            print("No NPZ files found, searching for JSON files...")
            json_files = list(self.data_dir.rglob("*.json"))
            print(f"Found {len(json_files)} JSON files")
            for json_file in json_files:
                identifier = json_file.stem
                landmark_files[identifier] = str(json_file)

        return landmark_files

    def match_annotations_to_files(self, annotations_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Match annotations to landmark files"""
        landmark_files = self.find_landmark_files()

        matched_paths = []
        matched_annotations = []

        for _, row in annotations_df.iterrows():
            identifier = row['id']
            annotation = row['annotation']

            if identifier in landmark_files:
                matched_paths.append(landmark_files[identifier])
                matched_annotations.append(annotation)
            else:
                matches = [path for id_key, path in landmark_files.items()
                           if identifier in id_key or id_key in identifier]
                if matches:
                    matched_paths.append(matches[0])
                    matched_annotations.append(annotation)
                else:
                    print(f"Warning: No landmark file found for {identifier}")

        file_type = "NPZ" if matched_paths and Path(matched_paths[0]).suffix == '.npz' else "JSON"
        print(f"Successfully matched {len(matched_paths)} annotations to {file_type} files")
        return matched_paths, matched_annotations

    def create_dataset(self, preprocessor: SignLanguagePreprocessor) -> PhoenixDataset:
        """Create Phoenix dataset"""
        annotations_df = self.load_annotations()
        file_paths, annotations = self.match_annotations_to_files(annotations_df)

        if not file_paths:
            raise ValueError("No files matched with annotations")

        dataset = PhoenixDataset(
            json_paths=file_paths,
            annotations=annotations,
            preprocessor=preprocessor
        )

        return dataset


def validate_dataset(dataset: PhoenixDataset) -> Dict:
    """Validate the dataset and return statistics"""
    stats = {
        'total_samples': len(dataset),
        'vocab_size': dataset.vocab_size,
        'annotation_lengths': [],
        'sequence_lengths': []
    }

    print("Validating dataset...")

    sample_size = min(10, len(dataset))

    for i in range(sample_size):
        try:
            sample = dataset[i]
            seq_length = sample['attention_mask'].sum().item()
            stats['sequence_lengths'].append(seq_length)

            ann_length = (sample['labels'] != 0).sum().item()
            stats['annotation_lengths'].append(ann_length)

            if i < 3:
                print(f"Sample {i}:")
                print(f"  Sequence length: {seq_length}")
                print(f"  Annotation length: {ann_length}")
                print(f"  Annotation: {sample['annotation']}")
                print()

        except Exception as e:
            print(f"Error validating sample {i}: {e}")

    if stats['sequence_lengths']:
        stats['avg_sequence_length'] = np.mean(stats['sequence_lengths'])
        stats['avg_annotation_length'] = np.mean(stats['annotation_lengths'])

    print(f"Dataset validation complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Average sequence length: {stats.get('avg_sequence_length', 'N/A')}")
    print(f"  Average annotation length: {stats.get('avg_annotation_length', 'N/A')}")

    return stats


def setup_wandb(config: Config):
    """Setup Weights & Biases logging"""
    wandb.init(
        project=config.logging.project_name,
        name=config.logging.experiment_name,
        config=config.to_dict(),
        tags=config.logging.tags,
        notes=config.logging.notes or f"Training LSTM model with batch size {config.training.batch_size}"
    )


def calculate_input_size(config: Config) -> int:
    """Calculate the correct input size based on preprocessor configuration"""
    preprocess_config = PreprocessingConfig(
        max_sequence_length=config.model.max_sequence_length,
        include_hands=config.preprocessing.include_hands,
        include_face=config.preprocessing.include_face,
        include_pose=config.preprocessing.include_pose,
        use_face_subset=config.preprocessing.use_face_subset,
        include_hand_confidence=config.preprocessing.include_hand_confidence,
        include_pose_visibility=config.preprocessing.include_pose_visibility
    )

    preprocessor = SignLanguagePreprocessor(preprocess_config)
    actual_input_size = preprocessor.feature_dims['total']

    print(f"Calculated input dimensions:")
    print(f"  - Hands: {preprocessor.feature_dims['hands']}")
    print(f"  - Face: {preprocessor.feature_dims['face']}")
    print(f"  - Pose: {preprocessor.feature_dims['pose']}")
    print(f"  - Total: {actual_input_size}")

    return actual_input_size


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options"""
    parser = argparse.ArgumentParser(
        description='Train LSTM model for sign language recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (primary method)
    parser.add_argument('--config', type=str,
                        help='Path to config YAML/JSON file (recommended)')

    # Data arguments (can override config)
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing landmark files')
    parser.add_argument('--annotations_path', type=str,
                        help='Path to annotations CSV/Excel file')
    parser.add_argument('--vocab_path', type=str,
                        help='Path to vocabulary file')

    # Model arguments
    parser.add_argument('--hidden_size', type=int,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--gradient_clip_norm', type=float,
                        help='Gradient clipping norm')
    parser.add_argument('--num_epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int,
                        help='Early stopping patience')

    # Sequence arguments
    parser.add_argument('--max_sequence_length', type=int,
                        help='Maximum sequence length')
    parser.add_argument('--max_annotation_length', type=int,
                        help='Maximum annotation length')

    # Output arguments
    parser.add_argument('--model_save_path', type=str,
                        help='Path to save trained model')
    parser.add_argument('--config_save_path', type=str, default='./loss14/config.yaml',
                        help='Path to save configuration')

    # WandB arguments
    parser.add_argument('--project_name', type=str,
                        help='WandB project name')
    parser.add_argument('--experiment_name', type=str,
                        help='WandB experiment name')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run WandB in offline mode')

    # System arguments
    parser.add_argument('--device', type=str,
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate the dataset without training')

    # Feature flags
    parser.add_argument('--no_faces', action='store_true',
                        help='Exclude face landmarks from features')

    return parser


def apply_args_to_config(config: Config, args: argparse.Namespace) -> Config:
    """Apply command-line arguments as overrides to config"""

    # Data overrides
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.annotations_path:
        config.data.annotations_path = args.annotations_path
    if args.vocab_path:
        config.data.vocab_path = args.vocab_path
    if args.model_save_path:
        config.data.model_save_path = args.model_save_path

    # Model overrides
    if args.hidden_size:
        config.model.hidden_size = args.hidden_size
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.dropout:
        config.model.dropout = args.dropout
    if args.max_sequence_length:
        config.model.max_sequence_length = args.max_sequence_length
        config.preprocessing.max_sequence_length = args.max_sequence_length
    if args.max_annotation_length:
        config.model.max_annotation_length = args.max_annotation_length

    # Training overrides
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.optimizer.learning_rate = args.learning_rate
    if args.weight_decay:
        config.training.optimizer.weight_decay = args.weight_decay
    if args.gradient_clip_norm:
        config.training.gradient_clip_norm = args.gradient_clip_norm
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.patience:
        config.training.patience = args.patience

    # Logging overrides
    if args.project_name:
        config.logging.project_name = args.project_name
    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
    if args.wandb_offline:
        config.logging.offline = True

    # Device override
    if args.device:
        config.device = args.device

    # Feature flags
    if args.no_faces:
        config.preprocessing.include_face = False

    return config


def main():
    """Main training function"""
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    # Load or create configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print("Creating default configuration")
        config = Config()

    # Apply command-line overrides
    config = apply_args_to_config(config, args)

    # Handle resume training
    checkpoint_vocab = None
    if args.resume:
        print("\n=== Loading Checkpoint for Resume ===")
        print(f"Checkpoint path: {args.resume}")

        if not Path(args.resume).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        checkpoint_config = checkpoint['config']
        checkpoint_vocab = checkpoint.get('vocab_size')

        print("Loaded checkpoint configuration:")
        print(f"  Input size: {checkpoint_config.input_size}")
        print(f"  Hidden size: {checkpoint_config.hidden_size}")
        print(f"  Vocabulary size: {checkpoint_vocab}")

        # Use checkpoint's model architecture
        config.model.input_size = checkpoint_config.input_size
        config.model.hidden_size = checkpoint_config.hidden_size
        config.model.num_layers = checkpoint_config.num_layers
        config.model.dropout = checkpoint_config.dropout

        print("\nWARNING: When resuming training, the model architecture and vocabulary")
        print("         are fixed from the checkpoint. The new dataset must be compatible.")

    # Calculate actual input size
    actual_input_size = calculate_input_size(config)
    config.model.input_size = actual_input_size

    print("\n=== Model Configuration ===")
    print(f"Input size: {config.model.input_size}")
    print(f"Hidden size: {config.model.hidden_size}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Max sequence length: {config.model.max_sequence_length}")
    print(f"Device: {config.device}")
    print()

    # Create output directories
    Path(config.data.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config_save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, args.config_save_path)
    print(f"Configuration saved to: {args.config_save_path}")

    print("\n=== LSTM Sign Language Training ===")
    print(f"Data directory: {config.data.data_dir}")
    print(f"Annotations file: {config.data.annotations_path}")
    print(f"Device: {config.device}")
    print()

    # Create dataset
    dataset_manager = PhoenixDatasetManager(
        data_dir=config.data.data_dir,
        annotations_path=config.data.annotations_path
    )

    preprocess_config = PreprocessingConfig(
        max_sequence_length=config.model.max_sequence_length,
        normalize_coordinates=config.preprocessing.normalize_coordinates,
        output_format=config.preprocessing.output_format,
        device=config.device,
        include_hand_confidence=config.preprocessing.include_hand_confidence,
        include_pose_visibility=config.preprocessing.include_pose_visibility,
        include_face=config.preprocessing.include_face,
    )
    preprocessor = SignLanguagePreprocessor(preprocess_config)

    # Verify input size for resume
    if args.resume:
        if preprocessor.feature_dims['total'] != config.model.input_size:
            print("\nERROR: Input size mismatch!")
            print(f"  Checkpoint expects: {config.model.input_size} features")
            print(f"  Current preprocessor produces: {preprocessor.feature_dims['total']} features")
            raise ValueError("Input size mismatch")

    print("Creating dataset...")
    dataset = dataset_manager.create_dataset(preprocessor)

    # Handle vocabulary for resume
    if args.resume and checkpoint_vocab:
        print("\nLoading vocabulary from checkpoint...")
        original_vocab_path = Path(args.resume).parent / "vocab.pkl"
        if original_vocab_path.exists():
            import pickle
            with open(original_vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)

            print(f"Applying checkpoint vocabulary ({checkpoint_vocab} glosses)...")
            if isinstance(vocab_data, dict):
                if 'gloss_to_idx' in vocab_data:
                    dataset.gloss_to_idx = vocab_data['gloss_to_idx']
                    dataset.idx_to_gloss = vocab_data['idx_to_gloss']
                else:
                    dataset.vocab = vocab_data
            dataset.vocab_size = checkpoint_vocab

    # Validate dataset
    dataset_stats = validate_dataset(dataset)

    if args.validate_only:
        print("Dataset validation complete. Exiting.")
        return

    # Setup WandB
    setup_wandb(config)

    wandb.log({
        'dataset/total_samples': dataset_stats['total_samples'],
        'dataset/vocab_size': dataset_stats['vocab_size'],
        'dataset/avg_sequence_length': dataset_stats.get('avg_sequence_length', 0),
        'dataset/avg_annotation_length': dataset_stats.get('avg_annotation_length', 0)
    })

    # Create trainer
    print("Initializing trainer...")
    feature_config = {'include_face': config.preprocessing.include_face}
    trainer = SignLanguageTrainer(config, feature_config)
    trainer.dataset = dataset
    trainer.train_loader, trainer.val_loader = trainer._create_data_loaders()

    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_model(args.resume)

    # Train
    print("Starting training...")
    trainer.train()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    trainer.evaluate_sample(0)

    # Save vocabulary
    dataset.save_vocabulary(config.data.vocab_path)

    print(f"\nTraining complete!")
    print(f"Model saved to: {config.data.model_save_path}")
    print(f"Vocabulary saved to: {config.data.vocab_path}")
    print(f"Configuration saved to: {args.config_save_path}")


if __name__ == "__main__":
    main()