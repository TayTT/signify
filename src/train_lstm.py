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
import json
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
from tqdm import tqdm

from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig, PhoenixDataset
from lstm_model import SignLanguageTrainer, ModelConfig


class PhoenixDatasetManager:
    """Enhanced Phoenix dataset manager with file-driven approach built-in"""

    def __init__(self, data_dir: str, annotations_path: str):
        self.data_dir = Path(data_dir)
        self.annotations_path = Path(annotations_path)

        # Results storage for file-driven approach
        self.valid_files = []
        self.invalid_files = []
        self.file_stats = {}

        # Validate paths
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

    def _validate_json_file(self, json_path: Path) -> Tuple[bool, Optional[str], int]:
        """
        Validate a single JSON file
        Returns: (is_valid, error_message, frame_count)
        """
        try:
            # Check file exists and has content
            if not json_path.exists():
                return False, "File does not exist", 0

            file_size = json_path.stat().st_size
            if file_size == 0:
                return False, "Empty file", 0

            # Try to load JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            if not isinstance(data, dict):
                return False, "Not a JSON object", 0

            if 'frames' not in data:
                return False, "Missing 'frames' key", 0

            frames = data['frames']
            if not frames:
                return False, "Empty frames", 0

            frame_count = len(frames)
            if frame_count < 5:  # Minimum viable sequence
                return False, f"Too few frames ({frame_count})", frame_count

            # Validate a sample frame has the correct structure
            sample_frame = list(frames.values())[0]

            # Check for main categories (not individual hand keys)
            required_categories = ['hands', 'face', 'pose']
            missing_categories = [cat for cat in required_categories if cat not in sample_frame]

            if missing_categories:
                return False, f"Missing categories: {missing_categories}", frame_count

            # Validate hands structure
            hands_data = sample_frame.get('hands', {})
            if not isinstance(hands_data, dict):
                return False, "Invalid hands data structure", frame_count

            # Optional: Check if hands actually contain some data
            # (You can remove this if you want to allow empty hands)
            has_left = 'left_hand' in hands_data and hands_data['left_hand']
            has_right = 'right_hand' in hands_data and hands_data['right_hand']
            if not (has_left or has_right):
                # This is just a warning, not a failure - some frames might not have hands
                pass

            # Validate face structure
            face_data = sample_frame.get('face', {})
            if not isinstance(face_data, dict):
                return False, "Invalid face data structure", frame_count

            # Validate pose structure
            pose_data = sample_frame.get('pose', {})
            if not isinstance(pose_data, dict):
                return False, "Invalid pose data structure", frame_count

            return True, None, frame_count

        except json.JSONDecodeError as e:
            return False, f"JSON decode error: {e}", 0
        except Exception as e:
            return False, f"Validation error: {e}", 0

    def find_json_files(self) -> Dict[str, str]:
        """Find and validate all JSON files with file-driven approach"""
        print("=" * 50)
        print("SCANNING AND VALIDATING JSON FILES")
        print("=" * 50)

        # Find all JSON files
        all_json_files = list(self.data_dir.rglob("*.json"))
        print(f"Found {len(all_json_files)} JSON files")

        valid_json_files = {}
        self.valid_files = []
        self.invalid_files = []

        # Validate each file with progress bar
        for json_file in tqdm(all_json_files, desc="Validating files"):
            is_valid, error_msg, frame_count = self._validate_json_file(json_file)

            if is_valid:
                # Store valid files
                identifier = json_file.stem
                valid_json_files[identifier] = str(json_file)
                self.valid_files.append({
                    'path': json_file,
                    'identifier': identifier,
                    'frame_count': frame_count
                })
            else:
                # Store invalid files for reporting
                self.invalid_files.append({
                    'path': json_file,
                    'error': error_msg,
                    'frame_count': frame_count
                })

        # Report results
        print(f"✓ Valid files: {len(valid_json_files)}")
        print(f"✗ Invalid files: {len(self.invalid_files)}")

        # Show sample invalid files
        if self.invalid_files:
            print("\nSample invalid files:")
            for i, invalid in enumerate(self.invalid_files[:5]):
                print(f"  {invalid['path'].name}: {invalid['error']}")
            if len(self.invalid_files) > 5:
                print(f"  ... and {len(self.invalid_files) - 5} more")

        # Calculate statistics
        if self.valid_files:
            total_frames = sum(f['frame_count'] for f in self.valid_files)
            avg_frames = total_frames / len(self.valid_files)
            self.file_stats = {
                'total_valid_files': len(self.valid_files),
                'total_invalid_files': len(self.invalid_files),
                'total_frames': total_frames,
                'avg_frames_per_file': avg_frames
            }

            print(f"\n=== FILE STATISTICS ===")
            print(f"Total frames: {total_frames:,}")
            print(f"Average frames per file: {avg_frames:.1f}")

        return valid_json_files

    def load_annotations(self) -> pd.DataFrame:
        """Load annotations from Excel or CSV file with enhanced error handling"""
        try:
            print(f"\nLoading annotations from: {self.annotations_path}")

            if self.annotations_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(self.annotations_path)
                print("✓ Loaded Excel file")
            else:
                # Try different separators for CSV files
                separators = ['|', ',', '\t', ';']
                df = None
                successful_separator = None

                print("Trying different separators...")
                for sep in separators:
                    try:
                        temp_df = pd.read_csv(self.annotations_path, sep=sep, nrows=5)
                        if len(temp_df.columns) > 1:
                            # Found a working separator, load full file
                            df = pd.read_csv(self.annotations_path, sep=sep)
                            successful_separator = sep
                            print(f"✓ Loaded CSV with separator '{sep}'")
                            break
                    except Exception:
                        continue

                if df is None:
                    raise ValueError("Could not parse CSV file with any separator")

            # Validate required columns
            if 'id' not in df.columns or 'annotation' not in df.columns:
                raise ValueError(f"Expected 'id' and 'annotation' columns, got: {list(df.columns)}")

            print(f"Loaded {len(df)} annotations")
            print(f"Columns: {list(df.columns)}")

            # Clean up the data
            df['id'] = df['id'].astype(str)
            df['annotation'] = df['annotation'].astype(str)

            return df

        except Exception as e:
            print(f"✗ Error loading annotations: {e}")
            print(f"File path: {self.annotations_path}")
            print(f"File exists: {self.annotations_path.exists()}")

            # Enhanced debugging info
            if self.annotations_path.exists():
                try:
                    print("File debugging info:")
                    with open(self.annotations_path, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline().strip() for _ in range(5)]

                    print(f"   File size: {self.annotations_path.stat().st_size} bytes")
                    print(f"   First 5 lines:")
                    for i, line in enumerate(first_lines):
                        line_preview = line[:100] + "..." if len(line) > 100 else line
                        print(f"     {i + 1}: {line_preview}")

                        # Check separators in first line
                        if i == 0:
                            separators = {'|': line.count('|'), ',': line.count(','),
                                          '\t': line.count('\t'), ';': line.count(';')}
                            print(f"     Separator counts: {separators}")

                except Exception as read_error:
                    print(f"   Could not read file for debugging: {read_error}")

            raise

    def match_annotations_to_json(self, annotations_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Match annotations to valid JSON files with enhanced reporting"""
        print("\n" + "=" * 50)
        print("MATCHING FILES TO ANNOTATIONS")
        print("=" * 50)

        # Get valid files (must call find_json_files first)
        if not self.valid_files:
            print("Warning: No valid files found. Did you call find_json_files()?")
            return [], []

        # Create lookup for valid files
        valid_files_lookup = {f['identifier']: str(f['path']) for f in self.valid_files}

        matched_json_paths = []
        matched_annotations = []
        unmatched_annotations = []

        for _, row in annotations_df.iterrows():
            identifier = str(row['id'])
            annotation = str(row['annotation'])

            # Try exact match first
            if identifier in valid_files_lookup:
                matched_json_paths.append(valid_files_lookup[identifier])
                matched_annotations.append(annotation)
            else:
                # Try partial matches
                matches = []
                for valid_id, valid_path in valid_files_lookup.items():
                    if identifier in valid_id or valid_id in identifier:
                        matches.append((valid_id, valid_path))

                if len(matches) == 1:
                    matched_json_paths.append(matches[0][1])
                    matched_annotations.append(annotation)
                else:
                    unmatched_annotations.append(identifier)

        # Report matching results
        print(f"✓ Successfully matched: {len(matched_json_paths)} pairs")
        print(f"✗ Unmatched annotations: {len(unmatched_annotations)}")

        if unmatched_annotations:
            print("\nSample unmatched annotations:")
            for i, unmatched in enumerate(unmatched_annotations[:5]):
                print(f"  {unmatched}")
            if len(unmatched_annotations) > 5:
                print(f"  ... and {len(unmatched_annotations) - 5} more")

        # Calculate success rates
        total_valid_files = len(self.valid_files)
        total_annotations = len(annotations_df)
        file_match_rate = (len(matched_json_paths) / total_valid_files) * 100 if total_valid_files > 0 else 0
        annotation_match_rate = (len(matched_json_paths) / total_annotations) * 100 if total_annotations > 0 else 0

        print(f"\n=== MATCHING STATISTICS ===")
        print(f"Valid files used: {len(matched_json_paths)}/{total_valid_files} ({file_match_rate:.1f}%)")
        print(f"Annotations matched: {len(matched_json_paths)}/{total_annotations} ({annotation_match_rate:.1f}%)")

        return matched_json_paths, matched_annotations

    def create_dataset(self, preprocessor: SignLanguagePreprocessor) -> PhoenixDataset:
        """Create dataset using integrated file-driven approach"""
        print("\n" + "=" * 60)
        print("CREATING DATASET WITH FILE-DRIVEN APPROACH")
        print("=" * 60)

        # Step 1: Find and validate JSON files (file-driven approach)
        json_files = self.find_json_files()

        if not json_files:
            raise ValueError("No valid JSON files found!")

        # Step 2: Load annotations
        annotations_df = self.load_annotations()

        if annotations_df.empty:
            raise ValueError("No annotations loaded!")

        # Step 3: Match valid files to annotations
        json_paths, annotations = self.match_annotations_to_json(annotations_df)

        if not json_paths:
            raise ValueError("No files could be matched to annotations!")

        # Step 4: Create dataset
        print(f"\n" + "=" * 50)
        print("CREATING PHOENIX DATASET")
        print("=" * 50)

        dataset = PhoenixDataset(
            json_paths=json_paths,
            annotations=annotations,
            preprocessor=preprocessor
        )

        # Final statistics
        print(f"\n=== FINAL DATASET SUMMARY ===")
        print(f"Total files scanned: {len(self.valid_files) + len(self.invalid_files)}")
        print(f"Valid files: {len(self.valid_files)}")
        print(f"Invalid files: {len(self.invalid_files)}")
        print(f"Successfully matched pairs: {len(json_paths)}")
        print(f"Dataset samples: {len(dataset)}")
        print(f"Vocabulary size: {dataset.vocab_size}")

        if self.file_stats:
            print(f"Total frames: {self.file_stats['total_frames']:,}")
            print(f"Average frames per file: {self.file_stats['avg_frames_per_file']:.1f}")

        # Success rate calculation
        if len(self.valid_files) > 0:
            success_rate = (len(json_paths) / len(self.valid_files)) * 100
            print(f"File utilization rate: {success_rate:.1f}%")

        return dataset

    def get_invalid_files_report(self) -> List[Dict]:
        """Get detailed report of invalid files for debugging"""
        return self.invalid_files

    def save_validation_report(self, output_path: str):
        """Save detailed validation report to JSON file"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_directory': str(self.data_dir),
            'annotations_path': str(self.annotations_path),
            'summary': {
                'total_files_scanned': len(self.valid_files) + len(self.invalid_files),
                'valid_files': len(self.valid_files),
                'invalid_files': len(self.invalid_files),
                **self.file_stats
            },
            'valid_files': [
                {
                    'path': str(f['path']),
                    'identifier': f['identifier'],
                    'frame_count': f['frame_count']
                }
                for f in self.valid_files
            ],
            'invalid_files': [
                {
                    'path': str(f['path']),
                    'error': f['error'],
                    'frame_count': f['frame_count']
                }
                for f in self.invalid_files
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Validation report saved to: {output_path}")

def create_config_from_args(args) -> ModelConfig:
    """Create model configuration from command line arguments"""

    # Calculate the correct input size with feature selection flags
    actual_input_size = calculate_input_size(args)

    # Create output directory and paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ModelConfig(
        # Use calculated input size
        input_size=actual_input_size,

        # Data paths
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,

        # Use output_dir for model and vocab paths
        model_save_path=str(output_dir / "lstm_sign2gloss.pth"),
        vocab_path=str(output_dir / "vocab.pkl"),

        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,

        # Model parameters (unidirectional LSTM for real-time)
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
    print(f"Feature configuration:")
    print(f"  - Include hands: {not getattr(args, 'no_hands', False)}")
    print(f"  - Include faces: {not getattr(args, 'no_faces', False)}")
    print(f"  - Include pose: {not getattr(args, 'no_pose', False)}")

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
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save model, vocabulary, and config files')

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

    # feature config
    feature_config = {
        'include_face': not getattr(args, 'no_faces', False)
    }

    # Create configuration
    config = create_config_from_args(args)

    # Create config save path in same output directory
    output_dir = Path(args.output_dir)
    config_save_path = output_dir / "config.yaml"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = asdict(config)

    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print(f"Configuration will be saved to: {config_save_path}")

    # Debug: Print configuration
    print("=== Model Configuration ===")
    print(f"Input size: {config.input_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max sequence length: {config.max_sequence_length}")
    print(f"Device: {config.device}")
    print()

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = asdict(config)
    config_dict['feature_config'] = feature_config

    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

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
    trainer = SignLanguageTrainer(config, feature_config)
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
    print(f"Configuration saved to: {args.config_save_path}") #TODO fix


if __name__ == "__main__":
    main()