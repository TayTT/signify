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
    python train_lstm.py --data_dir ./output --annotations_path ./phoenix_annotations.xlsx
"""

import os

import argparse

import pandas as pd
import torch
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import asdict
import yaml



from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig, PhoenixDataset
from lstm_model import SignLanguageTrainer, ModelConfig


class PhoenixDatasetManager:
    """Manages Phoenix dataset loading and processing"""

    def __init__(self, data_dir: str, annotations_path: str):
        self.data_dir = Path(data_dir)
        self.annotations_path = Path(annotations_path)

        # Validate paths
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

    def load_annotations(self) -> pd.DataFrame:
        """Load annotations from Excel or CSV file"""
        try:
            print(f"Loading annotations from: {self.annotations_path}")

            # Try to load file with automatic separator detection
            df = None
            successful_separator = None

            if self.annotations_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(self.annotations_path)
                print("Loaded Excel file")
            else:
                # For CSV files, try different separators
                separators = ['|', ',', '\t', ';']

                print("Trying different separators...")
                for sep in separators:
                    try:
                        test_df = pd.read_csv(self.annotations_path, sep=sep, nrows=3)
                        print(f"  Separator '{sep}': {len(test_df.columns)} columns -> {list(test_df.columns)}")

                        if len(test_df.columns) >= 4:  # Need at least 4 columns
                            df = pd.read_csv(self.annotations_path, sep=sep)
                            successful_separator = sep
                            print(f" Successfully loaded with separator '{sep}'")
                            break
                    except Exception as e:
                        print(f"  Separator '{sep}': Failed - {str(e)[:50]}...")
                        continue


            if df is None:
                raise ValueError("Failed to load annotations file")

            print(f"Initial shape: {df.shape}")
            print(f"Columns found: {list(df.columns)}")

            # Validate and fix column names
            required_columns = ['id', 'folder', 'signer', 'annotation']

            # Check if we have the exact column names
            if all(col in df.columns for col in required_columns):
                print(" All required columns found with exact names")
            else:
                print("Column names don't match exactly, attempting to map...")
                print(f"Expected: {required_columns}")
                print(f"Found: {list(df.columns)}")

                # Check if we have enough columns to map
                if len(df.columns) < 4:
                    raise ValueError(f"Insufficient columns: expected 4, found {len(df.columns)}")

                # Map first 4 columns to required names
                column_mapping = {}
                for i, req_col in enumerate(required_columns):
                    if i < len(df.columns):
                        old_col = df.columns[i]
                        column_mapping[old_col] = req_col

                print(f"Column mapping: {column_mapping}")
                df = df.rename(columns=column_mapping)

            # Keep only required columns and clean data
            df = df[required_columns].copy()

            # Clean the data
            initial_rows = len(df)
            df = df.dropna()  # Remove rows with missing values
            df['id'] = df['id'].astype(str).str.strip()
            df['annotation'] = df['annotation'].astype(str).str.strip()
            df['signer'] = df['signer'].astype(str).str.strip()
            df['folder'] = df['folder'].astype(str).str.strip()

            # Remove empty annotations
            df = df[df['annotation'].str.len() > 0]

            final_rows = len(df)
            if final_rows < initial_rows:
                print(f"️  Removed {initial_rows - final_rows} rows with missing/empty data")

            print(f" Final shape after cleaning: {df.shape}")
            print(f" Data summary:")
            print(f"   - Unique signers: {df['signer'].nunique()}")
            print(f"   - Unique IDs: {df['id'].nunique()}")
            print(f"   - Sample annotations:")
            for i, annotation in enumerate(df['annotation'].head(3)):
                print(f"     {i + 1}. {annotation}")

            return df

        except Exception as e:
            print(f" Error loading annotations: {e}")
            print(f"File path: {self.annotations_path}")
            print(f"File exists: {self.annotations_path.exists()}")

            # Enhanced debugging info
            if self.annotations_path.exists():
                try:
                    print(" File debugging info:")
                    with open(self.annotations_path, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline().strip() for _ in range(5)]

                    print(f"   File size: {self.annotations_path.stat().st_size} bytes")
                    print(f"   First 5 lines:")
                    for i, line in enumerate(first_lines):
                        line_preview = line[:100] + "..." if len(line) > 100 else line
                        print(f"     {i + 1}: {line_preview}")

                        # Check separators in first line
                        if i == 0:
                            separators = {'|': line.count('|'), ',': line.count(','), '\t': line.count('\t'),
                                          ';': line.count(';')}
                            print(f"     Separator counts: {separators}")

                except Exception as read_error:
                    print(f"   Could not read file for debugging: {read_error}")

            raise

    def find_json_files(self) -> Dict[str, str]:
        """Find all JSON files in the data directory"""
        json_files = {}

        # Search for JSON files
        for json_file in self.data_dir.rglob("*.json"):
            # Extract identifier from filename
            identifier = json_file.stem
            json_files[identifier] = str(json_file)

        print(f"Found {len(json_files)} JSON files")
        return json_files

    def match_annotations_to_json(self, annotations_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Match annotations to JSON files"""
        json_files = self.find_json_files()

        matched_json_paths = []
        matched_annotations = []

        for _, row in annotations_df.iterrows():
            identifier = row['id']
            annotation = row['annotation']

            # Try exact match first
            if identifier in json_files:
                matched_json_paths.append(json_files[identifier])
                matched_annotations.append(annotation)
            else:
                # Try partial matches
                matches = [path for id_key, path in json_files.items() if identifier in id_key or id_key in identifier]
                if matches:
                    matched_json_paths.append(matches[0])
                    matched_annotations.append(annotation)
                else:
                    print(f"Warning: No JSON file found for {identifier}")

        print(f"Successfully matched {len(matched_json_paths)} annotations to JSON files")
        return matched_json_paths, matched_annotations

    def create_dataset(self, preprocessor: SignLanguagePreprocessor) -> PhoenixDataset:
        """Create Phoenix dataset"""
        # Load annotations
        annotations_df = self.load_annotations()

        # Match annotations to JSON files
        json_paths, annotations = self.match_annotations_to_json(annotations_df)

        if not json_paths:
            raise ValueError("No JSON files matched with annotations")

        # Create dataset
        dataset = PhoenixDataset(
            json_paths=json_paths,
            annotations=annotations,
            preprocessor=preprocessor
        )

        return dataset


def create_config_from_args(args) -> ModelConfig:
    """Create model configuration from command line arguments"""

    # Calculate the correct input size
    actual_input_size = calculate_input_size(args)

    config = ModelConfig(
        # Use calculated input size instead of hardcoded value
        input_size=actual_input_size,

        # Data paths
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,
        model_save_path=args.model_save_path,
        vocab_path=args.vocab_path,

        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,

        # Model parameters
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=False,

        # Sequence parameters
        max_sequence_length=args.max_sequence_length,
        max_annotation_length=args.max_annotation_length,

        # WandB configuration
        project_name=args.project_name,
        experiment_name=args.experiment_name,

        # Device
        device=args.device
    )

    print(f"Model will be initialized with input_size={actual_input_size}")

    return config


def create_preprocessor_from_args(args) -> SignLanguagePreprocessor:
    """Create preprocessor with configuration from command line arguments"""
    preprocess_config = PreprocessingConfig(
        max_sequence_length=args.max_sequence_length,
        include_hands=not getattr(args, 'no_hands', False),
        include_face=not getattr(args, 'no_faces', False),
        include_pose=not getattr(args, 'no_pose', False),
        use_face_subset=True,
        include_hand_confidence=True,
        include_pose_visibility=True,
        normalize_coordinates=True,
        interpolate_missing=True
    )

    return SignLanguagePreprocessor(preprocess_config)

def calculate_input_size(config_args) -> int:
    """Calculate the correct input size based on preprocessor configuration"""
    preprocess_config = PreprocessingConfig(
        max_sequence_length=config_args.max_sequence_length,
        include_hands=True,
        include_face=not getattr(config_args, 'no_faces', False),
        include_pose=True,
        use_face_subset=True,
        include_hand_confidence=True,
        include_pose_visibility=True
    )

    preprocessor = SignLanguagePreprocessor(preprocess_config)
    actual_input_size = preprocessor.feature_dims['total']

    print(f"Calculated input dimensions:")
    print(f"  - Hands: {preprocessor.feature_dims['hands']}")
    print(f"  - Face: {preprocessor.feature_dims['face']}")
    print(f"  - Pose: {preprocessor.feature_dims['pose']}")
    print(f"  - Total: {actual_input_size}")

    return actual_input_size


def setup_wandb(config: ModelConfig, args):
    """Setup Weights & Biases logging"""
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config=asdict(config),
        tags=['lstm', 'real-time', 'sign-language', 'phoenix-dataset'],
        notes=f"Training real-time LSTM model on Phoenix dataset with {config.batch_size} batch size"
    )

    # Log additional information
    wandb.config.update({
        'data_dir': config.data_dir,
        'annotations_path': config.annotations_path,
        'resume_training': args.resume is not None
    })


def validate_dataset(dataset: PhoenixDataset) -> Dict:
    """Validate the dataset and return statistics"""
    stats = {
        'total_samples': len(dataset),
        'vocab_size': dataset.vocab_size,
        'annotation_lengths': [],
        'sequence_lengths': []
    }

    print("Validating dataset...")

    # Sample a few items to check
    sample_size = min(10, len(dataset))

    for i in range(sample_size):
        try:
            sample = dataset[i]

            # Check sequence length
            seq_length = sample['attention_mask'].sum().item()
            stats['sequence_lengths'].append(seq_length)

            # Check annotation length
            ann_length = (sample['labels'] != 0).sum().item()
            stats['annotation_lengths'].append(ann_length)

            if i < 3:  # Print first 3 samples
                print(f"Sample {i}:")
                print(f"  Sequence length: {seq_length}")
                print(f"  Annotation length: {ann_length}")
                print(f"  Annotation: {sample['annotation']}")
                print()

        except Exception as e:
            print(f"Error validating sample {i}: {e}")

    # Calculate statistics
    if stats['sequence_lengths']:
        stats['avg_sequence_length'] = np.mean(stats['sequence_lengths'])
        stats['avg_annotation_length'] = np.mean(stats['annotation_lengths'])

    print(f"Dataset validation complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Average sequence length: {stats.get('avg_sequence_length', 'N/A')}")
    print(f"  Average annotation length: {stats.get('avg_annotation_length', 'N/A')}")

    return stats


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train LSTM model for sign language recognition')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./output',
                        help='Directory containing JSON files')
    parser.add_argument('--annotations_path', type=str, default='./phoenix_annotations.xlsx',
                        help='Path to annotations Excel file')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl',
                        help='Path to vocabulary file')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # Feature arguments
    parser.add_argument('--no-faces', action='store_true',
                        help='Exclude face landmarks from training (ignore face data)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Sequence arguments
    parser.add_argument('--max_sequence_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--max_annotation_length', type=int, default=50,
                        help='Maximum annotation length')

    # Output arguments
    parser.add_argument('--model_save_path', type=str, default='./models/lstm_sign2gloss.pth',
                        help='Path to save trained model')
    parser.add_argument('--config_save_path', type=str, default='./models/config.yaml',
                        help='Path to save configuration')

    # WandB arguments
    parser.add_argument('--project_name', type=str, default='sign-language-lstm',
                        help='WandB project name')
    parser.add_argument('--experiment_name', type=str, default='lstm-sign2gloss',
                        help='WandB experiment name')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run WandB in offline mode')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate the dataset without training')

    args = parser.parse_args()

    # Set up WandB mode
    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    # Create configuration
    config = create_config_from_args(args)

    # Debug: Print configuration
    print("=== Model Configuration ===")
    print(f"Input size: {config.input_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max sequence length: {config.max_sequence_length}")
    print(f"Device: {config.device}")
    print()

    # Create output directories
    Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config_save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(args.config_save_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)

    print("=== LSTM Sign Language Training ===")
    print(f"Data directory: {config.data_dir}")
    print(f"Annotations file: {config.annotations_path}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max sequence length: {config.max_sequence_length}")
    print()

    # Initialize dataset manager
    dataset_manager = PhoenixDatasetManager(
        data_dir=config.data_dir,
        annotations_path=config.annotations_path
    )

    # Create preprocessor
    preprocess_config = PreprocessingConfig(
        max_sequence_length=config.max_sequence_length,
        normalize_coordinates=False,
        output_format="tensor",
        device=config.device,
        include_face=not getattr(args, 'no_faces', False),
        use_face_subset=True,
        include_hand_confidence=True,
        include_pose_visibility=True
    )
    preprocessor = SignLanguagePreprocessor(preprocess_config)

    # Create dataset
    print("Creating dataset...")
    dataset = dataset_manager.create_dataset(preprocessor)

    # Validate dataset
    dataset_stats = validate_dataset(dataset)

    if args.validate_only:
        print("Dataset validation complete. Exiting.")
        return

    # Setup WandB
    setup_wandb(config, args)

    # Log dataset statistics
    wandb.log({
        'dataset/total_samples': dataset_stats['total_samples'],
        'dataset/vocab_size': dataset_stats['vocab_size'],
        'dataset/avg_sequence_length': dataset_stats.get('avg_sequence_length', 0),
        'dataset/avg_annotation_length': dataset_stats.get('avg_annotation_length', 0)
    })

    # Initialize trainer
    print("Initializing trainer...")
    trainer = SignLanguageTrainer(config)
    trainer.dataset = dataset  # Override dataset
    trainer.train_loader, trainer.val_loader = trainer._create_data_loaders()

    # Resume training if checkpoint provided
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_model(args.resume)

    # Start training
    print("Starting training...")
    trainer.train()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    trainer.evaluate_sample(0)

    # Save final vocabulary
    dataset.save_vocabulary(config.vocab_path)

    print(f"\nTraining complete!")
    print(f"Model saved to: {config.model_save_path}")
    print(f"Vocabulary saved to: {config.vocab_path}")
    print(f"Configuration saved to: {args.config_save_path}")


if __name__ == "__main__":
    main()