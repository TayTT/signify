#!/usr/bin/env python3
"""
Transfer Learning Script for Cross-Dataset Sign Language Training
WITH COMPREHENSIVE WANDB TRACKING

Use case: Pre-train on ASL Citizen, then fine-tune on Phoenix Weather

This script:
1. Loads pre-trained model from one dataset
2. Replaces output layer for new vocabulary
3. Optionally freezes encoder layers
4. Fine-tunes on new dataset
5. Tracks everything in WandB

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
import numpy as np
from pathlib import Path
from dataclasses import asdict
import wandb
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_jsons import SignLanguagePreprocessor, PreprocessingConfig, PhoenixDataset
from lstm_model import SignLanguageLSTM, SignLanguageTrainer, ModelConfig
from train_lstm import PhoenixDatasetManager, validate_dataset


def load_pretrained_model(checkpoint_path: str):
    """Load pre-trained model and extract components"""
    print(f"\n=== Loading Pre-trained Model ===")
    print(f"Checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    pretrained_config = checkpoint['config']
    pretrained_vocab_size = checkpoint['vocab_size']

    # Infer preprocessing config from input size
    input_size = pretrained_config.input_size

    print(f"Pre-trained model configuration:")
    print(f"  Input size: {pretrained_config.input_size}")
    print(f"  Hidden size: {pretrained_config.hidden_size}")
    print(f"  Num layers: {pretrained_config.num_layers}")
    print(f"  Vocabulary size: {pretrained_vocab_size}")
    print(f"  Max sequence length: {pretrained_config.max_sequence_length}")

    print(f"\n=== COMPLETE PRE-TRAINED MODEL CONFIG ===")
    if hasattr(pretrained_config, '__dict__'):
        for key, value in pretrained_config.__dict__.items():
            print(f"{key}: {value}")
    else:
        print(pretrained_config)

    print(f"\n=== CHECKPOINT CONTENTS ===")
    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Vocab size in checkpoint: {checkpoint.get('vocab_size', 'N/A')}")

    # Log pre-trained model info to WandB
    if wandb.run is not None:
        wandb.log({
            "pretrained/input_size": pretrained_config.input_size,
            "pretrained/hidden_size": pretrained_config.hidden_size,
            "pretrained/num_layers": pretrained_config.num_layers,
            "pretrained/vocab_size": pretrained_vocab_size,
            "pretrained/dropout": pretrained_config.dropout,
        })

        # Log training history if available
        if 'train_losses' in checkpoint:
            pretrained_history = {
                "pretrained/final_train_loss": checkpoint['train_losses'][-1] if checkpoint['train_losses'] else None,
                "pretrained/final_val_loss": checkpoint['val_losses'][-1] if checkpoint['val_losses'] else None,
                "pretrained/best_val_loss": checkpoint.get('best_val_loss', None),
                "pretrained/num_epochs_trained": len(checkpoint['train_losses'])
            }
            wandb.log(pretrained_history)

        # Extract preprocessing config if available
        preprocessing_info = {
            'include_hands': getattr(pretrained_config, 'include_hands', True),
            'include_face': getattr(pretrained_config, 'include_face', False),
            'include_pose': getattr(pretrained_config, 'include_pose', True),
            'use_face_subset': getattr(pretrained_config, 'use_face_subset', True),
            'include_hand_confidence': getattr(pretrained_config, 'include_hand_confidence', False),  # NPZ files don't contain confidence scores
            'include_pose_visibility': getattr(pretrained_config, 'include_pose_visibility', False),
        }

        print(f"\nDetected preprocessing config:")
        for key, val in preprocessing_info.items():
            print(f"  {key}: {val}")

        return checkpoint, pretrained_config, preprocessing_info


def create_target_dataset(data_dir: str, annotations_path: str,
                          pretrained_config: ModelConfig, device: str,
                          preprocessing_info: dict = None):
    """Create dataset for target domain (new sign language)"""
    print(f"\n=== Creating Target Dataset ===")
    print(f"Data directory: {data_dir}")
    print(f"Annotations: {annotations_path}")

    # Initialize dataset manager
    dataset_manager = PhoenixDatasetManager(
        data_dir=data_dir,
        annotations_path=annotations_path
    )

    # Calculate expected face landmarks from pre-trained model
    include_hands = preprocessing_info.get('include_hands', True) if preprocessing_info else True
    include_face = preprocessing_info.get('include_face', False) if preprocessing_info else False
    include_pose = preprocessing_info.get('include_pose', True) if preprocessing_info else True
    use_face_subset = preprocessing_info.get('use_face_subset', True) if preprocessing_info else True
    include_hand_confidence = preprocessing_info.get('include_hand_confidence',
                                                     True) if preprocessing_info else False
    include_pose_visibility = preprocessing_info.get('include_pose_visibility',
                                                     False) if preprocessing_info else False

    face_subset_indices = None
    include_face=True
    if include_face and use_face_subset:
        # Calculate what face landmarks we need
        if include_hand_confidence:
            hands_features = 128
        else:
            hands_features = 126

        # Account for pose if included
        if include_pose:
                pose_features = 27  # 9 pose landmarks × 3
        else:
            pose_features = 0

        face_features = pretrained_config.input_size - hands_features - pose_features
        expected_face_landmarks = face_features // 3

        print(
            f"Calculation: {pretrained_config.input_size} - {hands_features} hands - {pose_features} pose = {face_features} face")
        print(f"Expected face landmarks: {expected_face_landmarks}")

        # ACTUALLY CREATE THE LIST!
        full_face_list = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]
        face_subset_indices = full_face_list[:expected_face_landmarks]

        print(f"Created face_subset_indices: {face_subset_indices}")

    # Create config ONCE with all parameters
    preprocess_config = PreprocessingConfig(
        max_sequence_length=pretrained_config.max_sequence_length,
        include_hands=include_hands,
        include_face=include_face,
        include_pose=include_pose,
        use_face_subset=use_face_subset,
        include_hand_confidence=include_hand_confidence,
        include_pose_visibility=include_pose_visibility,
        face_subset_indices=face_subset_indices,
        normalize_coordinates=False,
        output_format="tensor",
        device=device
    )

    # Before creating preprocessor
    print(f"\n=== BEFORE creating preprocessor ===")
    print(f"face_subset_indices we're passing: {face_subset_indices}")
    print(f"Length: {len(face_subset_indices) if face_subset_indices else 'None'}")

    preprocessor = SignLanguagePreprocessor(preprocess_config)

    print(f"\n=== FEATURE BREAKDOWN ===")
    print(f"Hands: {preprocessor.feature_dims.get('hands', 0)}")
    print(f"Face: {preprocessor.feature_dims.get('face', 0)}")
    print(f"Pose: {preprocessor.feature_dims.get('pose', 0)}")
    print(f"Total: {preprocessor.feature_dims['total']}")
    print(f"\nNeeded: {pretrained_config.input_size}")
    print(f"Difference: {pretrained_config.input_size - preprocessor.feature_dims['total']}")

    # Calculate what each should be
    print(f"\n=== CONFIGURATION ===")
    print(f"include_hands: {preprocess_config.include_hands}")
    print(f"include_hand_confidence: {preprocess_config.include_hand_confidence}")
    print(f"include_face: {preprocess_config.include_face}")
    print(f"use_face_subset: {preprocess_config.use_face_subset}")
    if preprocess_config.face_subset_indices:
        print(f"face_subset_indices count: {len(preprocess_config.face_subset_indices)}")
    print(f"include_pose: {preprocess_config.include_pose}")
    print(f"include_pose_visibility: {preprocess_config.include_pose_visibility}")

    print(f"\n=== preprocessing_info ===")
    print(preprocessing_info)


    # Verify input size matches
    actual_input_size = preprocessor.feature_dims['total']
    if actual_input_size != pretrained_config.input_size:
        print(f"\nWARNING: Input size mismatch!")
        print(f"  Pre-trained model expects: {pretrained_config.input_size}")
        print(f"  Current preprocessor produces: {actual_input_size}")
        print(f"\nAttempting to adjust preprocessor configuration...")

        raise ValueError(
            f"Input size mismatch. Please adjust PreprocessingConfig to produce "
            f"{pretrained_config.input_size} features instead of {actual_input_size}"
        )

    # Create dataset with new vocabulary
    dataset = dataset_manager.create_dataset(preprocessor)

    print(f"Target dataset created:")
    print(f"  Samples: {len(dataset)}")
    print(f"  New vocabulary size: {dataset.vocab_size}")

    # Log target dataset info to WandB
    if wandb.run is not None:
        wandb.log({
            "target/num_samples": len(dataset),
            "target/vocab_size": dataset.vocab_size,
            "target/input_size": actual_input_size,
        })

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

    # Track parameter counts
    transferred_params = 0
    skipped_params = 0
    new_params = 0

    for key in pretrained_state:
        # Transfer all layers except the classifier (output layer)
        if key in new_state and 'classifier' not in key:
            if pretrained_state[key].shape == new_state[key].shape:
                new_state[key] = pretrained_state[key]
                transferred_layers.append(key)
                transferred_params += pretrained_state[key].numel()
            else:
                print(f"  Shape mismatch for {key}: "
                      f"{pretrained_state[key].shape} vs {new_state[key].shape}")
                skipped_layers.append(key)
                skipped_params += pretrained_state[key].numel()
        elif 'classifier' in key:
            # Skip classifier layers as they have different vocabulary size
            skipped_layers.append(key)
            skipped_params += pretrained_state[key].numel()
            print(f"  Skipping {key} (output layer for different vocabulary)")

    # Count new parameters (classifier)
    for key in new_state:
        if 'classifier' in key:
            new_params += new_state[key].numel()

    # Load the transferred weights
    new_model.load_state_dict(new_state)

    print(f"\nTransfer summary:")
    print(f"  Transferred: {len(transferred_layers)} layers ({transferred_params:,} parameters)")
    print(f"  Skipped: {len(skipped_layers)} layers ({skipped_params:,} parameters)")
    print(f"  New (randomly initialized): {new_params:,} parameters")
    print(f"  Transfer ratio: {transferred_params / (transferred_params + new_params) * 100:.1f}%")

    # Log transfer statistics to WandB
    if wandb.run is not None:
        wandb.log({
            "transfer/num_layers_transferred": len(transferred_layers),
            "transfer/num_layers_skipped": len(skipped_layers),
            "transfer/params_transferred": transferred_params,
            "transfer/params_new": new_params,
            "transfer/transfer_ratio": transferred_params / (transferred_params + new_params) * 100,
        })

        # Create a table showing layer transfer details
        transfer_table = wandb.Table(
            columns=["Layer", "Status", "Parameters", "Shape"]
        )

        for key in transferred_layers[:10]:  # Show first 10
            transfer_table.add_data(
                key,
                "Transferred",
                pretrained_state[key].numel(),
                str(pretrained_state[key].shape)
            )

        for key in [k for k in new_state.keys() if 'classifier' in k]:
            transfer_table.add_data(
                key,
                "New (Random)",
                new_state[key].numel(),
                str(new_state[key].shape)
            )

        wandb.log({"transfer/layer_details": transfer_table})

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

        frozen_layers = []
        trainable_layers = []

        for name, param in new_model.named_parameters():
            # Freeze everything except classifier
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
                frozen_layers.append(name)
            else:
                trainable_params += param.numel()
                trainable_layers.append(name)

        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params / (frozen_params + trainable_params) * 100:.1f}%")

        # Log freeze statistics to WandB
        if wandb.run is not None:
            wandb.log({
                "freeze/frozen_params": frozen_params,
                "freeze/trainable_params": trainable_params,
                "freeze/trainable_ratio": trainable_params / (frozen_params + trainable_params) * 100,
                "freeze/num_frozen_layers": len(frozen_layers),
                "freeze/num_trainable_layers": len(trainable_layers),
            })

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


def setup_enhanced_wandb(config: ModelConfig, args, pretrained_checkpoint: dict,
                         target_dataset, transferred_params: dict):
    """Setup WandB with comprehensive configuration"""

    # Initialize WandB
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config=asdict(config),
        tags=['transfer-learning', 'cross-dataset', 'fine-tuning'],
        notes=f"Transfer learning: freeze_encoder={args.freeze_encoder}, "
              f"pretrained_vocab={pretrained_checkpoint['vocab_size']}, "
              f"target_vocab={target_dataset.vocab_size}"
    )

    # Log comprehensive configuration
    wandb.config.update({
        # Source model info
        'pretrained_model_path': args.pretrained_model,
        'pretrained_vocab_size': pretrained_checkpoint['vocab_size'],
        'pretrained_best_val_loss': pretrained_checkpoint.get('best_val_loss', None),

        # Target dataset info
        'target_data_dir': args.data_dir,
        'target_annotations': args.annotations_path,
        'target_vocab_size': target_dataset.vocab_size,
        'target_num_samples': len(target_dataset),

        # Transfer learning settings
        'freeze_encoder': args.freeze_encoder,
        'unfreeze_after_epoch': args.unfreeze_after_epoch if args.freeze_encoder else None,

        # Training strategy
        'training_strategy': 'frozen_encoder' if args.freeze_encoder else 'full_finetuning',
        'initial_learning_rate': args.learning_rate,
        'reduced_learning_rate': args.learning_rate * 0.1 if args.unfreeze_after_epoch > 0 else None,
    })

    print(f"WandB initialized: {wandb.run.url}")


def log_training_phase(phase_name: str, epoch_start: int, epoch_end: int,
                       learning_rate: float, frozen: bool):
    """Log training phase information to WandB"""
    if wandb.run is not None:
        wandb.log({
            f"phase/{phase_name}/start_epoch": epoch_start,
            f"phase/{phase_name}/end_epoch": epoch_end,
            f"phase/{phase_name}/learning_rate": learning_rate,
            f"phase/{phase_name}/encoder_frozen": frozen,
        })
        print(f"\nLogged training phase '{phase_name}' to WandB")


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
    print("WITH COMPREHENSIVE WANDB TRACKING")
    print("=" * 70)

    # Initialize WandB early for logging during setup
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        tags=['transfer-learning', 'setup'],
    )

    # Step 1: Load pre-trained model
    pretrained_checkpoint, pretrained_config, preprocessing_info = load_pretrained_model(args.pretrained_model)

    # Step 2: Create target dataset
    target_dataset = create_target_dataset(
        args.data_dir,
        args.annotations_path,
        pretrained_config,
        args.device,
        preprocessing_info
    )

    # Validate target dataset
    print("\n=== Validating Target Dataset ===")
    dataset_stats = validate_dataset(target_dataset)

    # Log dataset statistics to WandB
    if wandb.run is not None:
        wandb.log({
            'dataset/total_samples': dataset_stats['total_samples'],
            'dataset/vocab_size': dataset_stats['vocab_size'],
            'dataset/avg_sequence_length': dataset_stats.get('avg_sequence_length', 0),
            'dataset/avg_annotation_length': dataset_stats.get('avg_annotation_length', 0)
        })

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

    # Step 6: Update WandB config with all information
    wandb.config.update({
        'pretrained_model_path': args.pretrained_model,
        'pretrained_vocab_size': pretrained_checkpoint['vocab_size'],
        'target_vocab_size': target_dataset.vocab_size,
        'freeze_encoder': args.freeze_encoder,
        'unfreeze_after_epoch': args.unfreeze_after_epoch,
    })

    # Step 7: Setup trainer
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

    # Step 8: Start training with phase tracking
    print(f"\n=== Starting Fine-Tuning ===")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Freeze encoder: {args.freeze_encoder}")

    if args.freeze_encoder and args.unfreeze_after_epoch > 0:
        print(f"Will unfreeze encoder after epoch {args.unfreeze_after_epoch}")

    # Custom training loop with optional unfreezing
    if args.freeze_encoder and args.unfreeze_after_epoch > 0:
        # Phase 1: Train with frozen encoder
        original_epochs = config.num_epochs
        config.num_epochs = args.unfreeze_after_epoch

        log_training_phase(
            "phase_1_frozen",
            epoch_start=0,
            epoch_end=config.num_epochs,
            learning_rate=config.learning_rate,
            frozen=True
        )

        print(f"\nPhase 1: Training classifier only for {config.num_epochs} epochs")
        trainer.train()

        # Unfreeze encoder
        print(f"\n=== Unfreezing Encoder ===")
        for param in trainer.model.parameters():
            param.requires_grad = True

        # Re-initialize optimizer with all parameters
        new_lr = config.learning_rate * 0.1
        trainer.optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=new_lr,
            weight_decay=config.weight_decay
        )
        print(f"Reduced learning rate to {new_lr} for full fine-tuning")

        # Log unfreezing event
        wandb.log({
            "phase/unfreeze_epoch": args.unfreeze_after_epoch,
            "phase/new_learning_rate": new_lr,
        })

        # Phase 2: Continue training
        remaining_epochs = original_epochs - args.unfreeze_after_epoch
        config.num_epochs = remaining_epochs

        log_training_phase(
            "phase_2_unfrozen",
            epoch_start=args.unfreeze_after_epoch,
            epoch_end=original_epochs,
            learning_rate=new_lr,
            frozen=False
        )

        print(f"\nPhase 2: Fine-tuning full model for {config.num_epochs} more epochs")
        trainer.best_val_loss = float('inf')  # Reset for phase 2
        trainer.train()
    else:
        # Standard training (single phase)
        log_training_phase(
            "single_phase",
            epoch_start=0,
            epoch_end=config.num_epochs,
            learning_rate=config.learning_rate,
            frozen=args.freeze_encoder
        )
        trainer.train()

    # Step 9: Save results
    print(f"\n=== Saving Results ===")
    trainer.save_model()
    target_dataset.save_vocabulary(config.vocab_path)

    # Save model as WandB artifact
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"{config.experiment_name}-model",
            type="model",
            description=f"Transfer learning model: {args.pretrained_model} -> {args.data_dir}",
            metadata={
                "pretrained_vocab_size": pretrained_checkpoint['vocab_size'],
                "target_vocab_size": target_dataset.vocab_size,
                "freeze_encoder": args.freeze_encoder,
                "final_val_loss": trainer.best_val_loss,
            }
        )
        artifact.add_file(config.model_save_path)
        artifact.add_file(config.vocab_path)
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)
        print("Model saved as WandB artifact")

    print(f"\nTransfer learning complete!")
    print(f"Model saved to: {config.model_save_path}")
    print(f"Vocabulary saved to: {config.vocab_path}")
    print(f"Config saved to: {config_path}")

    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    trainer.evaluate_sample(0)

    # Log final summary
    if wandb.run is not None:
        wandb.summary.update({
            "final/best_val_loss": trainer.best_val_loss,
            "final/total_epochs": len(trainer.train_losses),
            "final/model_path": config.model_save_path,
            "final/pretrained_source": args.pretrained_model,
        })

    wandb.finish()
    print(f"\nWandB run complete: {wandb.run.url if wandb.run else 'N/A'}")


if __name__ == "__main__":
    main()