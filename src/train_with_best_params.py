#!/usr/bin/env python3
"""
Train LSTM Model with Best Parameters from Optuna Study

This script loads the best hyperparameters found by Optuna and trains
a final model with full epochs.

Usage:
    python train_with_best_params.py --best_config ./optuna_studies/best_config.yaml

    # With overrides:
    python train_with_best_params.py --best_config ./optuna_studies/best_config.yaml --num_epochs 100
"""

import os
import sys
import argparse
import torch
import wandb
from pathlib import Path
from dataclasses import asdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, load_config, save_config
from preprocess_jsons import SignLanguagePreprocessor
from lstm_model import SignLanguageLSTM, SignLanguageTrainer
from train_lstm import PhoenixDatasetManager, validate_dataset


def create_optimizer_from_config(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    """Create optimizer based on configuration"""
    opt_config = config.training.optimizer
    optimizer_name = opt_config.name.lower()
    lr = opt_config.learning_rate
    weight_decay = opt_config.weight_decay

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.eps
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.eps
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_config.momentum,
            weight_decay=weight_decay,
            nesterov=opt_config.nesterov
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            alpha=0.99,
            eps=opt_config.eps
        )
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}', using AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    return optimizer


def create_scheduler_from_config(optimizer: torch.optim.Optimizer, config: Config):
    """Create scheduler based on configuration"""
    sched_config = config.training.scheduler
    scheduler_type = sched_config.name.lower()
    num_epochs = config.training.num_epochs

    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config.factor,
            patience=sched_config.patience,
            min_lr=sched_config.min_lr
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=sched_config.min_lr
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.step_size,
            gamma=sched_config.gamma
        )
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=sched_config.gamma or 0.95
        )
    elif scheduler_type == 'lambda':
        warmup_steps = sched_config.warmup_steps
        total_steps = sched_config.total_steps

        def lr_lambda_func(step):
            if step < warmup_steps:
                return max(0.1, step / warmup_steps)
            else:
                import math
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                progress = max(0.0, min(1.0, progress))
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_func)
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

    # Required: config file from Optuna
    parser.add_argument('--best_config', type=str, required=True,
                        help='Path to best_config.yaml from Optuna study')

    # Optional: data paths (override config)
    parser.add_argument('--data_dir', type=str,
                        help='Override data directory')
    parser.add_argument('--annotations_path', type=str,
                        help='Override annotations path')

    # Optional: training overrides
    parser.add_argument('--num_epochs', type=int,
                        help='Override number of epochs')
    parser.add_argument('--model_save_path', type=str,
                        help='Override model save path')
    parser.add_argument('--experiment_name', type=str,
                        help='Override experiment name')
    parser.add_argument('--device', type=str,
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

    config = load_config(args.best_config)

    # Display loaded parameters
    print("\nBest Hyperparameters:")
    print("-" * 40)
    print("Model Architecture:")
    print(f"  hidden_size: {config.model.hidden_size}")
    print(f"  num_layers: {config.model.num_layers}")
    print(f"  dropout: {config.model.dropout}")
    print(f"  bidirectional: {config.model.bidirectional}")

    print("\nTraining Parameters:")
    print(f"  batch_size: {config.training.batch_size}")
    print(f"  learning_rate: {config.training.optimizer.learning_rate}")
    print(f"  weight_decay: {config.training.optimizer.weight_decay}")
    print(f"  gradient_clip_norm: {config.training.gradient_clip_norm}")

    print("\nOptimizer Configuration:")
    print(f"  optimizer: {config.training.optimizer.name}")
    if config.training.optimizer.name.lower() in ['adam', 'adamw']:
        print(f"  beta1: {config.training.optimizer.beta1}")
        print(f"  beta2: {config.training.optimizer.beta2}")
    elif config.training.optimizer.name.lower() == 'sgd':
        print(f"  momentum: {config.training.optimizer.momentum}")
        print(f"  nesterov: {config.training.optimizer.nesterov}")

    print("\nScheduler Configuration:")
    print(f"  scheduler: {config.training.scheduler.name}")
    if config.training.scheduler.name.lower() == 'plateau':
        print(f"  factor: {config.training.scheduler.factor}")
        print(f"  patience: {config.training.scheduler.patience}")
    elif config.training.scheduler.name.lower() == 'step':
        print(f"  step_size: {config.training.scheduler.step_size}")
        print(f"  gamma: {config.training.scheduler.gamma}")

    # Apply overrides
    if args.data_dir:
        config.data.data_dir = args.data_dir
        print(f"\nOverride: data_dir -> {args.data_dir}")
    if args.annotations_path:
        config.data.annotations_path = args.annotations_path
        print(f"Override: annotations_path -> {args.annotations_path}")
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
        print(f"Override: num_epochs -> {args.num_epochs}")
    if args.model_save_path:
        config.data.model_save_path = args.model_save_path
        print(f"Override: model_save_path -> {args.model_save_path}")
    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
        print(f"Override: experiment_name -> {args.experiment_name}")
    else:
        config.logging.experiment_name = f"{config.logging.experiment_name}_best_final"
    if args.device:
        config.device = args.device
        print(f"Override: device -> {args.device}")

    print("\n" + "=" * 60)
    print("Starting Training with Best Parameters")
    print("=" * 60)

    # Load dataset
    dataset_manager = PhoenixDatasetManager(
        data_dir=config.data.data_dir,
        annotations_path=config.data.annotations_path
    )

    preprocessor = SignLanguagePreprocessor(config.preprocessing)

    print("\nCreating dataset...")
    dataset = dataset_manager.create_dataset(preprocessor)

    # Update input size
    actual_input_size = preprocessor.feature_dims['total']
    config.model.input_size = actual_input_size

    print(f"Dataset: {len(dataset)} samples, vocab size: {dataset.vocab_size}")
    print(f"Input size: {actual_input_size}")

    # Validate dataset
    dataset_stats = validate_dataset(dataset)

    # Initialize WandB
    wandb.init(
        project=config.logging.project_name,
        name=config.logging.experiment_name,
        config=config.to_dict(),
        tags=["best_params", "final_training"]
    )

    # Create trainer
    print("\nInitializing trainer...")
    feature_config = {'include_face': config.preprocessing.include_face}
    trainer = SignLanguageTrainer(config, feature_config)
    trainer.dataset = dataset
    trainer.train_loader, trainer.val_loader = trainer._create_data_loaders()

    # Replace optimizer and scheduler with configured versions
    trainer.optimizer = create_optimizer_from_config(trainer.model, config)
    trainer.scheduler = create_scheduler_from_config(trainer.optimizer, config)

    print("\n" + "=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Model: {config.model.num_layers} layers, {config.model.hidden_size} hidden units")
    print(f"Bidirectional: {config.model.bidirectional}")
    print(f"Optimizer: {config.training.optimizer.name}")
    print(f"Initial LR: {config.training.optimizer.learning_rate}")
    print(f"Scheduler: {config.training.scheduler.name}")
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
    vocab_path = Path(config.data.model_save_path).parent / "vocab.pkl"
    dataset.save_vocabulary(str(vocab_path))

    # Save final config
    config_save_path = Path(config.data.model_save_path).parent / "final_config.yaml"
    save_config(config, config_save_path)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Model saved to: {config.data.model_save_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Configuration saved to: {config_save_path}")
    print(f"WandB run: {wandb.run.url if wandb.run else 'N/A'}")
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    main()