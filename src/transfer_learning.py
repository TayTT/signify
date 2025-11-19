#!/usr/bin/env python3
"""
Transfer Learning Script for Cross-Dataset Sign Language Training

Use case: Pre-train on ASL Citizen, then fine-tune on Phoenix Weather

This script:
1. Loads pre-trained model from one dataset
2. Replaces output layer for new vocabulary
3. Optionally freezes encoder layers
4. Fine-tunes on new dataset

Usage:
    python transfer_learning.py \
        --pretrained_model ./asl_citizen_model/lstm_sign2gloss.pth \
        --data_dir ./phoenix_data \
        --annotations_path ./phoenix_annotations.csv \
        --output_dir ./phoenix_finetuned \
        --freeze_encoder \
        --num_epochs 50
"""

import os
import sys
import argparse
import torch
import yaml
import pickle
from pathlib import Path
from dataclasses import asdict
import wandb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig, PhoenixDataset
from lstm_model import SignLanguageLSTM, SignLanguageTrainer, ModelConfig
from train_lstm import PhoenixDatasetManager, validate_dataset, setup_wandb


def load_pretrained_model(checkpoint_path: str):
    """Load pre-trained model and extract components"""
    print(f"\n=== Loading Pre-trained Model ===")
    print(f"Checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    pretrained_config = checkpoint['config']
    pretrained_vocab_size = checkpoint['vocab_size']

    print(f"Pre-trained model configuration:")
    print(f"  Input size: {pretrained_config.input_size}")
    print(f"  Hidden size: {pretrained_config.hidden_size}")
    print(f"  Num layers: {pretrained_config.num_layers}")
    print(f"  Vocabulary size: {pretrained_vocab_size}")
    print(f"  Max sequence length: {pretrained_config.max_sequence_length}")

    return checkpoint, pretrained_config


def create_target_dataset(data_dir: str, annotations_path: str,
                          pretrained_config: ModelConfig, device: str):
    """Create dataset for target domain (new sign language)"""
    print(f"\n=== Creating Target Dataset ===")
    print(f"Data directory: {data_dir}")
    print(f"Annotations: {annotations_path}")

    # Initialize dataset manager
    dataset_manager = PhoenixDatasetManager(
        data_dir=data_dir,
        annotations_path=annotations_path
    )

    # Create preprocessor matching pre-trained model's input size
    preprocess_config = PreprocessingConfig(
        max_sequence_length=pretrained_config.max_sequence_length,
        normalize_coordinates=False,
        output_format="tensor",
        device=device
    )
    preprocessor = SignLanguagePreprocessor(preprocess_config)

    # Verify input size matches
    actual_input_size = preprocessor.feature_dims['total']
    if actual_input_size != pretrained_config.input_size:
        print(f"\nWARNING: Input size mismatch!")
        print(f"  Pre-trained model expects: {pretrained_config.input_size}")
        print(f"  Current preprocessor produces: {actual_input_size}")
        print(f"\nAttempting to adjust preprocessor configuration...")

        # You might need to adjust PreprocessingConfig parameters here
        # For example, exclude certain landmarks if input is too large
        raise ValueError(
            f"Input size mismatch. Please adjust PreprocessingConfig to produce "
            f"{pretrained_config.input_size} features instead of {actual_input_size}"
        )

    # Create dataset with new vocabulary
    dataset = dataset_manager.create_dataset(preprocessor)

    print(f"Target dataset created:")
    print(f"  Samples: {len(dataset)}")
    print(f"  New vocabulary size: {dataset.vocab_size}")

    return dataset


def transfer_weights(pretrained_checkpoint: dict, new_model: SignLanguageLSTM,
                     freeze_encoder: bool = False):
    """
    Transfer weights from pre-trained model to new model

    Args:
        pretrained_checkpoint: Loaded checkpoint from pre-trained model
        new_model: New model with different output size
        freeze_encoder: Whether to freeze LSTM encoder layers
    """
    print(f"\n=== Transferring Weights ===")

    pretrained_state = pretrained_checkpoint['model_state_dict']
    new_state = new_model.state_dict()

    transferred_layers = []
    skipped_layers = []

    for key in pretrained_state:
        # Transfer all layers except the classifier (output layer)
        if key in new_state and 'classifier' not in key:
            if pretrained_state[key].shape == new_state[key].shape:
                new_state[key] = pretrained_state[key]
                transferred_layers.append(key)
            else:
                print(f"  Shape mismatch for {key}: "
                      f"{pretrained_state[key].shape} vs {new_state[key].shape}")
                skipped_layers.append(key)
        elif 'classifier' in key:
            # Skip classifier layers as they have different vocabulary size
            skipped_layers.append(key)
            print(f"  Skipping {key} (output layer for different vocabulary)")

    # Load the transferred weights
    new_model.load_state_dict(new_state)

    print(f"\nTransfer summary:")
    print(f"  Transferred: {len(transferred_layers)} layers")
    print(f"  Skipped: {len(skipped_layers)} layers")

    if transferred_layers:
        print(f"\n  Key transferred layers:")
        for layer in transferred_layers[:5]:
            print(f"    - {layer}")
        if len(transferred_layers) > 5:
            print(f"    ... and {len(transferred_layers) - 5} more")

    # Freeze encoder layers if requested
    if freeze_encoder:
        print(f"\n=== Freezing Encoder Layers ===")
        frozen_params = 0
        trainable_params = 0

        for name, param in new_model.named_parameters():
            # Freeze everything except classifier
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                trainable_params += param.numel()

        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Ratio: {trainable_params / (frozen_params + trainable_params) * 100:.1f}% trainable")

    return new_model


def create_transfer_config(pretrained_config: ModelConfig, args) -> ModelConfig:
    """Create config for transfer learning"""
    config = ModelConfig(
        # Keep architecture from pre-trained model
        input_size=pretrained_config.input_size,
        hidden_size=pretrained_config.hidden_size,
        num_layers=pretrained_config.num_layers,
        dropout=pretrained_config.dropout,
        bidirectional=pretrained_config.bidirectional,
        max_sequence_length=pretrained_config.max_sequence_length,
        max_annotation_length=pretrained_config.max_annotation_length,

        # New dataset paths
        data_dir=args.data_dir,
        annotations_path=args.annotations_path,
        vocab_path=os.path.join(args.output_dir, "vocab.pkl"),
        model_save_path=os.path.join(args.output_dir, "lstm_transferred.pth"),

        # Training parameters (can be different for fine-tuning)
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,

        # WandB
        project_name=args.project_name,
        experiment_name=args.experiment_name,

        # Device
        device=args.device
    )

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Transfer learning for cross-dataset sign language training'
    )

    # Pre-trained model
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pre-trained model checkpoint (.pth)')

    # Target dataset
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing target dataset JSON files')
    parser.add_argument('--annotations_path', type=str, required=True,
                        help='Path to target dataset annotations')
    parser.add_argument('--output_dir', type=str, default='./transferred_model',
                        help='Directory to save transferred model')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for fine-tuning')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (typically lower for fine-tuning)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Transfer learning options
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze LSTM encoder layers (only train classifier)')
    parser.add_argument('--unfreeze_after_epoch', type=int, default=0,
                        help='Unfreeze encoder after N epochs (0=never, if frozen)')

    # WandB
    parser.add_argument('--project_name', type=str, default='sign-language-transfer',
                        help='WandB project name')
    parser.add_argument('--experiment_name', type=str, default='transfer-asl-to-phoenix',
                        help='WandB experiment name')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run WandB in offline mode')

    # System
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    args = parser.parse_args()

    # Set up WandB mode
    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRANSFER LEARNING FOR SIGN LANGUAGE RECOGNITION")
    print("=" * 70)

    # Step 1: Load pre-trained model
    pretrained_checkpoint, pretrained_config = load_pretrained_model(args.pretrained_model)

    # Step 2: Create target dataset
    target_dataset = create_target_dataset(
        args.data_dir,
        args.annotations_path,
        pretrained_config,
        args.device
    )

    # Validate target dataset
    print("\n=== Validating Target Dataset ===")
    dataset_stats = validate_dataset(target_dataset)

    # Step 3: Create config for transfer learning
    config = create_transfer_config(pretrained_config, args)

    # Save config
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    print(f"\nSaved config to: {config_path}")

    # Step 4: Initialize new model with target vocabulary size
    print(f"\n=== Creating New Model ===")
    print(f"Architecture: Same as pre-trained model")
    print(f"New vocabulary size: {target_dataset.vocab_size}")

    new_model = SignLanguageLSTM(config, target_dataset.vocab_size)

    # Step 5: Transfer weights
    new_model = transfer_weights(
        pretrained_checkpoint,
        new_model,
        freeze_encoder=args.freeze_encoder
    )

    # Step 6: Setup trainer
    print(f"\n=== Setting Up Trainer ===")
    trainer = SignLanguageTrainer(config)
    trainer.model = new_model.to(trainer.device)
    trainer.dataset = target_dataset
    trainer.train_loader, trainer.val_loader = trainer._create_data_loaders()

    # Re-initialize optimizer with correct parameters
    if args.freeze_encoder:
        # Only optimize classifier parameters
        trainable_params = filter(lambda p: p.requires_grad, trainer.model.parameters())
        trainer.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        print("Optimizer configured for classifier-only training")
    else:
        # Optimize all parameters (full fine-tuning)
        print("Optimizer configured for full fine-tuning")

    # Setup WandB
    print(f"\n=== Setting Up WandB ===")
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config=asdict(config),
        tags=['transfer-learning', 'cross-dataset', 'fine-tuning'],
        notes=f"Transfer from pre-trained model, freeze_encoder={args.freeze_encoder}"
    )

    wandb.config.update({
        'pretrained_model': args.pretrained_model,
        'freeze_encoder': args.freeze_encoder,
        'target_vocab_size': target_dataset.vocab_size,
        'pretrained_vocab_size': pretrained_checkpoint['vocab_size']
    })

    # Step 7: Start training
    print(f"\n=== Starting Fine-Tuning ===")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Freeze encoder: {args.freeze_encoder}")

    if args.freeze_encoder and args.unfreeze_after_epoch > 0:
        print(f"Will unfreeze encoder after epoch {args.unfreeze_after_epoch}")

    # Custom training loop with optional unfreezing
    if args.freeze_encoder and args.unfreeze_after_epoch > 0:
        # Train with frozen encoder first
        original_epochs = config.num_epochs
        config.num_epochs = args.unfreeze_after_epoch
        print(f"\nPhase 1: Training classifier only for {config.num_epochs} epochs")
        trainer.train()

        # Unfreeze encoder
        print(f"\n=== Unfreezing Encoder ===")
        for param in trainer.model.parameters():
            param.requires_grad = True

        # Re-initialize optimizer with all parameters
        trainer.optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=config.learning_rate * 0.1,  # Lower LR for unfrozen encoder
            weight_decay=config.weight_decay
        )
        print(f"Reduced learning rate to {config.learning_rate * 0.1} for full fine-tuning")

        # Continue training
        config.num_epochs = original_epochs - args.unfreeze_after_epoch
        print(f"\nPhase 2: Fine-tuning full model for {config.num_epochs} more epochs")
        trainer.best_val_loss = float('inf')  # Reset for phase 2
        trainer.train()
    else:
        # Standard training
        trainer.train()

    # Step 8: Save results
    print(f"\n=== Saving Results ===")
    trainer.save_model()
    target_dataset.save_vocabulary(config.vocab_path)

    print(f"\nTransfer learning complete!")
    print(f"Model saved to: {config.model_save_path}")
    print(f"Vocabulary saved to: {config.vocab_path}")
    print(f"Config saved to: {config_path}")

    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    trainer.evaluate_sample(0)

    wandb.finish()


if __name__ == "__main__":
    main()