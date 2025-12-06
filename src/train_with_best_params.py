#!/usr/bin/env python3
"""
Train LSTM Model with Best Parameters from Optuna Study

This script loads the best hyperparameters found by Optuna and trains
a final model with full epochs.

Usage:
    python train_with_best_params.py --data_dir ./output --annotations_path ./annotations.csv --best_config ./optuna_studies/best_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import wandb
from pathlib import Path
from dataclasses import replace, asdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig
from lstm_model import SignLanguageLSTM, SignLanguageTrainer, ModelConfig
from train_lstm import PhoenixDatasetManager


def load_config_from_yaml(yaml_path: str) -> ModelConfig:
    """Load ModelConfig from YAML file"""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig(**config_dict)


def create_optimizer_from_config(model, config_dict):
    """Create optimizer based on saved configuration"""
    optimizer_name = config_dict.get('optimizer_name', 'adamw')
    lr = config_dict.get('learning_rate', 1e-4)
    weight_decay = config_dict.get('weight_decay', 1e-4)

    if optimizer_name == 'adam':
        beta1 = config_dict.get('beta1', 0.9)
        beta2 = config_dict.get('beta2', 0.999)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=1e-8
        )
    elif optimizer_name == 'adamw':
        beta1 = config_dict.get('beta1', 0.9)
        beta2 = config_dict.get('beta2', 0.999)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=1e-8
        )
    elif optimizer_name == 'sgd':
        momentum = config_dict.get('momentum', 0.9)
        nesterov = config_dict.get('nesterov', True)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            alpha=0.99,
            eps=1e-8
        )
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}', using AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    return optimizer


def create_scheduler_from_config(optimizer, config_dict, num_epochs):
    """Create scheduler based on saved configuration"""
    scheduler_type = config_dict.get('scheduler_type', 'plateau')

    if scheduler_type == 'plateau':
        factor = config_dict.get('scheduler_factor', 0.7)
        patience = config_dict.get('scheduler_patience', 8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=1e-7
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-7
        )
    elif scheduler_type == 'step':
        step_size = config_dict.get('step_size', 10)
        gamma = config_dict.get('gamma', 0.7)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    else:
        print(f"Warning: Unknown scheduler '{scheduler_type}', using ReduceLROnPlateau")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=8,
            min_lr=1e-7
        )

    return scheduler


def main():
    parser = argparse.ArgumentParser(description='Train with best Optuna parameters')

    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed landmark files')
    parser.add_argument('--annotations_path', type=str, required=True,
                        help='Path to annotations CSV/Excel file')
    parser.add_argument('--best_config', type=str, required=True,
                        help='Path to best_config.yaml from Optuna study')

    # Optional overrides
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override number of epochs (default: use value from config)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Override model save path')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Override experiment name')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda/cpu)')

    # WandB settings
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run WandB in offline mode')

    args = parser.parse_args()

    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    # Load best configuration
    print("=" * 60)
    print("Loading Best Configuration from Optuna Study")
    print("=" * 60)
    print(f"Config file: {args.best_config}")

    if not Path(args.best_config).exists():
        raise FileNotFoundError(f"Config file not found: {args.best_config}")

    # Load config
    with open(args.best_config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Display loaded parameters
    print("\nBest Hyperparameters:")
    print("-" * 40)
    print("Model Architecture:")
    print(f"  hidden_size: {config_dict['hidden_size']}")
    print(f"  num_layers: {config_dict['num_layers']}")
    print(f"  dropout: {config_dict['dropout']}")
    print(f"  bidirectional: {config_dict['bidirectional']}")

    print("\nTraining Parameters:")
    print(f"  batch_size: {config_dict['batch_size']}")
    print(f"  learning_rate: {config_dict['learning_rate']}")
    print(f"  weight_decay: {config_dict['weight_decay']}")
    print(f"  gradient_clip_norm: {config_dict.get('gradient_clip_norm', 1.0)}")

    print("\nOptimizer Configuration:")
    optimizer_name = config_dict.get('optimizer_name', 'adamw')
    print(f"  optimizer: {optimizer_name}")
    if optimizer_name in ['adam', 'adamw']:
        print(f"  beta1: {config_dict.get('beta1', 0.9)}")
        print(f"  beta2: {config_dict.get('beta2', 0.999)}")
    elif optimizer_name == 'sgd':
        print(f"  momentum: {config_dict.get('momentum', 0.9)}")
        print(f"  nesterov: {config_dict.get('nesterov', True)}")

    print("\nScheduler Configuration:")
    scheduler_type = config_dict.get('scheduler_type', 'plateau')
    print(f"  scheduler: {scheduler_type}")
    if scheduler_type == 'plateau':
        print(f"  factor: {config_dict.get('scheduler_factor', 0.7)}")
        print(f"  patience: {config_dict.get('scheduler_patience', 8)}")
    elif scheduler_type == 'step':
        print(f"  step_size: {config_dict.get('step_size', 10)}")
        print(f"  gamma: {config_dict.get('gamma', 0.7)}")

    # Apply overrides
    config_dict['data_dir'] = args.data_dir
    config_dict['annotations_path'] = args.annotations_path

    if args.num_epochs:
        config_dict['num_epochs'] = args.num_epochs
        print(f"\nOverride: num_epochs -> {args.num_epochs}")

    if args.model_save_path:
        config_dict['model_save_path'] = args.model_save_path
        print(f"Override: model_save_path -> {args.model_save_path}")

    if args.experiment_name:
        config_dict['experiment_name'] = args.experiment_name
        print(f"Override: experiment_name -> {args.experiment_name}")
    else:
        config_dict['experiment_name'] = f"{config_dict['experiment_name']}_best_final"

    if args.device:
        config_dict['device'] = args.device
        print(f"Override: device -> {args.device}")

    # Create ModelConfig
    config = ModelConfig(**config_dict)

    print("\n" + "=" * 60)
    print("Starting Training with Best Parameters")
    print("=" * 60)

    # Load dataset
    dataset_manager = PhoenixDatasetManager(
        data_dir=config.data_dir,
        annotations_path=config.annotations_path
    )

    preprocess_config = PreprocessingConfig(
        max_sequence_length=config.max_sequence_length,
        normalize_coordinates=False,
        output_format="tensor",
        device=config.device,
        include_hand_confidence=False,
        include_pose_visibility=False
    )
    preprocessor = SignLanguagePreprocessor(preprocess_config)

    print("\nCreating dataset...")
    dataset = dataset_manager.create_dataset(preprocessor)

    # Update input size
    actual_input_size = preprocessor.feature_dims['total']
    config = replace(config, input_size=actual_input_size)

    print(f"Dataset: {len(dataset)} samples, vocab size: {dataset.vocab_size}")
    print(f"Input size: {actual_input_size}")

    # Initialize WandB
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config={**asdict(config), **config_dict},
        tags=["best_params", "final_training"]
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SignLanguageTrainer(config)
    trainer.dataset = dataset
    trainer.train_loader, trainer.val_loader = trainer._create_data_loaders()

    # Replace optimizer and scheduler with optimized versions
    trainer.optimizer = create_optimizer_from_config(trainer.model, config_dict)
    trainer.scheduler = create_scheduler_from_config(
        trainer.optimizer,
        config_dict,
        config.num_epochs
    )

    # Set gradient clipping
    trainer.config = replace(
        trainer.config,
        gradient_clip_norm=config_dict.get('gradient_clip_norm', 1.0)
    )

    print("\n" + "=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Model: {config.num_layers} layers, {config.hidden_size} hidden units")
    print(f"Bidirectional: {config.bidirectional}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Initial LR: {config.learning_rate}")
    print(f"Scheduler: {scheduler_type}")
    print("=" * 60 + "\n")

    # Start training
    print("Starting training...")
    trainer.train()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    trainer.evaluate_sample(0)

    # Save vocabulary
    vocab_path = Path(config.model_save_path).parent / "vocab.pkl"
    dataset.save_vocabulary(str(vocab_path))

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Model saved to: {config.model_save_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"WandB run: {wandb.run.url if wandb.run else 'N/A'}")
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    main()