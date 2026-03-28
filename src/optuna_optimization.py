#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for LSTM Sign Language Recognition

This script uses Optuna to find optimal hyperparameters including:
- Model architecture (hidden_size, num_layers, dropout, bidirectional)
- Optimizer choice (Adam, AdamW, SGD, RMSprop)
- Training parameters (learning_rate, batch_size, weight_decay)
- Scheduler parameters

Usage:
    # Using config file:
    python optuna_optimization.py --config config.yaml --n_trials 50

    # Using command-line arguments:
    python optuna_optimization.py --data_dir ./output --annotations_path ./annotations.csv
"""

import os
import sys
import argparse
import torch
import optuna
from optuna.trial import TrialState
import wandb
from pathlib import Path
import yaml
from typing import Dict, Any
from dataclasses import asdict
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    Config, load_config, save_config,
    TrainingConfig, OptimizerConfig, SchedulerConfig
)
from preprocess_jsons  import SignLanguagePreprocessor, PhoenixDataset
from lstm_model import SignLanguageLSTM
from train_lstm import PhoenixDatasetManager


class OptunaOptimizer:
    """Manages Optuna hyperparameter optimization"""

    def __init__(self, base_config: Config, args: argparse.Namespace):
        self.base_config = base_config
        self.args = args
        self.best_value = float('inf')
        self.best_params = None
        self.device = torch.device(base_config.device)

        # Load dataset once to avoid reloading
        print("Loading dataset for optimization...")
        self.dataset_manager = PhoenixDatasetManager(
            data_dir=base_config.data.data_dir,
            annotations_path=base_config.data.annotations_path
        )

        self.preprocessor = SignLanguagePreprocessor(base_config.preprocessing)

        # Store input size
        self.input_size = self.preprocessor.feature_dims['total']

        # Load dataset
        self.dataset = self.dataset_manager.create_dataset(self.preprocessor)
        print(f"Dataset loaded: {len(self.dataset)} samples, vocab size: {self.dataset.vocab_size}")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial"""

        # Model Architecture
        hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512, 768])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        bidirectional = trial.suggest_categorical('bidirectional', [False])

        # Training Parameters
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        # Optimizer Choice
        optimizer_name = trial.suggest_categorical(
            'optimizer',
            ['adam', 'adamw', 'sgd', 'rmsprop']
        )

        # Optimizer-specific parameters
        if optimizer_name == 'sgd':
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            nesterov = trial.suggest_categorical('nesterov', [True, False])
        else:
            momentum = None
            nesterov = None

        if optimizer_name in ['adam', 'adamw']:
            beta1 = trial.suggest_float('beta1', 0.85, 0.95)
            beta2 = trial.suggest_float('beta2', 0.95, 0.999)
        else:
            beta1 = None
            beta2 = None

        # Scheduler Parameters
        scheduler_type = trial.suggest_categorical(
            'scheduler_type',
            ['plateau', 'cosine', 'step', 'exponential']
        )

        scheduler_factor = None
        scheduler_patience = None
        step_size = None
        gamma = None

        if scheduler_type == 'plateau':
            scheduler_factor = trial.suggest_float('scheduler_factor', 0.5, 0.9)
            scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)
        elif scheduler_type == 'step':
            step_size = trial.suggest_int('step_size', 5, 20)
            gamma = trial.suggest_float('gamma', 0.5, 0.9)

        # Gradient Clipping
        gradient_clip = trial.suggest_float('gradient_clip', 0.5, 5.0)

        return {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer_name': optimizer_name,
            'momentum': momentum,
            'nesterov': nesterov,
            'beta1': beta1,
            'beta2': beta2,
            'scheduler_type': scheduler_type,
            'scheduler_factor': scheduler_factor,
            'scheduler_patience': scheduler_patience,
            'step_size': step_size,
            'gamma': gamma,
            'gradient_clip': gradient_clip,
        }

    def create_optimizer(self, model: torch.nn.Module, params: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer based on trial parameters"""
        optimizer_name = params['optimizer_name']
        lr = params['learning_rate']
        weight_decay = params['weight_decay']

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(params['beta1'], params['beta2']),
                eps=1e-8
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(params['beta1'], params['beta2']),
                eps=1e-8
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=params['momentum'],
                weight_decay=weight_decay,
                nesterov=params['nesterov']
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
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def create_scheduler(self, optimizer: torch.optim.Optimizer, params: Dict[str, Any]):
        """Create learning rate scheduler based on trial parameters"""
        scheduler_type = params['scheduler_type']

        if scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=params['scheduler_factor'],
                patience=params['scheduler_patience'],
                min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.max_epochs_per_trial,
                eta_min=1e-7
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=params['step_size'],
                gamma=params['gamma']
            )
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=5
            )

        return scheduler

    def _create_data_loaders(self, batch_size: int):
        """Create train and validation data loaders"""
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn
        )

        return train_loader, val_loader

    def _collate_fn(self, batch):
        """Collate function for data loader"""
        sequences = [item['sequence'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        annotations = [item['annotation'] for item in batch]

        batch_max_seq = max(seq.shape[0] for seq in sequences)
        max_label_len = max(label.shape[0] for label in labels)

        padded_seqs = []
        padded_masks = []
        padded_labels = []

        for seq, mask, label in zip(sequences, attention_masks, labels):
            if seq.shape[0] < batch_max_seq:
                seq_pad = torch.zeros(batch_max_seq - seq.shape[0], seq.shape[1])
                seq = torch.cat([seq, seq_pad], dim=0)
                mask_pad = torch.zeros(batch_max_seq - mask.shape[0])
                mask = torch.cat([mask, mask_pad], dim=0)

            if label.shape[0] < max_label_len:
                label_pad = torch.zeros(max_label_len - label.shape[0], dtype=label.dtype)
                label = torch.cat([label, label_pad], dim=0)

            padded_seqs.append(seq)
            padded_masks.append(mask)
            padded_labels.append(label)

        return {
            'sequence': torch.stack(padded_seqs),
            'attention_mask': torch.stack(padded_masks),
            'labels': torch.stack(padded_labels),
            'annotation': annotations,
            'metadata': [{}] * len(batch)
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        params = self.suggest_hyperparameters(trial)

        # Create config for this trial
        trial_config = Config()
        trial_config.model.input_size = self.input_size
        trial_config.model.hidden_size = params['hidden_size']
        trial_config.model.num_layers = params['num_layers']
        trial_config.model.dropout = params['dropout']
        trial_config.model.bidirectional = params['bidirectional']
        trial_config.model.use_ctc = self.base_config.model.use_ctc
        trial_config.preprocessing.max_sequence_length = self.base_config.preprocessing.max_sequence_length
        trial_config.device = str(self.device)

        # Initialize WandB for this trial
        wandb.init(
            project=self.base_config.logging.project_name,
            name=f"trial_{trial.number}",
            config=params,
            reinit=True,
            tags=['optuna', 'hyperparameter-search']
        )

        try:
            # Create model
            model = SignLanguageLSTM(trial_config, self.dataset.vocab_size).to(self.device)

            # Create optimizer and scheduler
            optimizer = self.create_optimizer(model, params)
            scheduler = self.create_scheduler(optimizer, params)

            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(params['batch_size'])

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 5

            for epoch in range(self.args.max_epochs_per_trial):
                # Train
                model.train()
                train_loss = 0

                for batch in train_loader:
                    sequences = batch['sequence'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    optimizer.zero_grad()
                    outputs = model(sequences, attention_mask, labels)
                    loss = outputs['loss']
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validate
                model.eval()
                val_loss = 0

                with torch.no_grad():
                    for batch in val_loader:
                        sequences = batch['sequence'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = model(sequences, attention_mask, labels)
                        val_loss += outputs['loss'].item()

                val_loss /= len(val_loader)

                # Update scheduler
                if params['scheduler_type'] == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                # Log to WandB
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

                # Report to Optuna for pruning
                trial.report(val_loss, epoch)

                if trial.should_prune():
                    wandb.finish()
                    raise optuna.TrialPruned()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            wandb.finish()
            return best_val_loss

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            wandb.finish()
            raise

    def run_optimization(self, n_trials: int = 50):
        """Run the optimization study"""
        study_dir = Path(self.args.study_dir)
        study_dir.mkdir(parents=True, exist_ok=True)

        storage = f"sqlite:///{study_dir}/optuna_study.db"

        study = optuna.create_study(
            study_name=self.args.study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        )

        print(f"\n{'=' * 60}")
        print(f"Starting Optuna Optimization Study")
        print(f"{'=' * 60}")
        print(f"Study name: {self.args.study_name}")
        print(f"Number of trials: {n_trials}")
        print(f"Max epochs per trial: {self.args.max_epochs_per_trial}")
        print(f"Storage: {storage}")
        print(f"{'=' * 60}\n")

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=self.args.timeout,
            catch=(Exception,)
        )

        # Print results
        print("\n" + "=" * 60)
        print("Optimization Complete")
        print("=" * 60)

        print(f"\nNumber of finished trials: {len(study.trials)}")

        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        print(f"  Pruned trials: {len(pruned_trials)}")
        print(f"  Completed trials: {len(complete_trials)}")

        if len(complete_trials) > 0:
            print("\nBest trial:")
            trial = study.best_trial

            print(f"  Value (val_loss): {trial.value:.4f}")
            print(f"  Params:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

            # Save best parameters
            best_params_path = study_dir / "best_params.yaml"
            with open(best_params_path, 'w') as f:
                yaml.dump(trial.params, f, default_flow_style=False)
            print(f"\nBest parameters saved to: {best_params_path}")

            # Create best config
            best_config = self._create_config_from_params(trial.params)
            best_config_path = study_dir / "best_config.yaml"
            save_config(best_config, best_config_path)
            print(f"Best configuration saved to: {best_config_path}")

            # Generate visualization plots
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                fig.write_html(study_dir / "optimization_history.html")

                fig = optuna.visualization.plot_param_importances(study)
                fig.write_html(study_dir / "param_importances.html")

                fig = optuna.visualization.plot_slice(study)
                fig.write_html(study_dir / "param_slice.html")

                print(f"\nVisualization plots saved to: {study_dir}/")
            except ImportError:
                print("\nInstall plotly to generate visualization plots: pip install plotly")

        return study

    def _create_config_from_params(self, params: Dict[str, Any]) -> Config:
        """Create Config from best parameters"""
        config = Config()

        params = {**params}
        if 'optimizer' in params and 'optimizer_name' not in params:
            params['optimizer_name'] = params['optimizer']

        for key in ['beta1', 'beta2', 'momentum', 'nesterov',
                    'scheduler_factor', 'scheduler_patience', 'step_size', 'gamma']:
            params.setdefault(key, None)

        # Model config
        config.model.input_size = self.input_size
        config.model.hidden_size = params['hidden_size']
        config.model.num_layers = params['num_layers']
        config.model.dropout = params['dropout']
        config.model.bidirectional = params['bidirectional']
        config.model.use_ctc = self.base_config.model.use_ctc
        config.preprocessing.max_sequence_length = self.base_config.preprocessing.max_sequence_length

        # Training config
        config.training.batch_size = params['batch_size']
        config.training.gradient_clip_norm = params['gradient_clip']

        # Optimizer config
        config.training.optimizer.name = params['optimizer_name']
        config.training.optimizer.learning_rate = params['learning_rate']
        config.training.optimizer.weight_decay = params['weight_decay']
        if params['beta1']:
            config.training.optimizer.beta1 = params['beta1']
        if params['beta2']:
            config.training.optimizer.beta2 = params['beta2']
        if params['momentum']:
            config.training.optimizer.momentum = params['momentum']
        if params['nesterov'] is not None:
            config.training.optimizer.nesterov = params['nesterov']

        # Scheduler config
        config.training.scheduler.name = params['scheduler_type']
        if params['scheduler_factor']:
            config.training.scheduler.factor = params['scheduler_factor']
        if params['scheduler_patience']:
            config.training.scheduler.patience = params['scheduler_patience']
        if params['step_size']:
            config.training.scheduler.step_size = params['step_size']
        if params['gamma']:
            config.training.scheduler.gamma = params['gamma']

        # Data config
        config.data.data_dir = self.base_config.data.data_dir
        config.data.annotations_path = self.base_config.data.annotations_path
        config.data.vocab_path = self.base_config.data.vocab_path

        # Logging config
        config.logging.project_name = self.base_config.logging.project_name
        config.logging.experiment_name = f"{self.base_config.logging.experiment_name}_best"

        # Device
        config.device = str(self.device)

        return config


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization for LSTM')

    # Config file
    parser.add_argument('--config', type=str,
                        help='Path to config YAML/JSON file')

    # Data paths (if not using config file)
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing processed landmark files')
    parser.add_argument('--annotations_path', type=str,
                        help='Path to annotations CSV/Excel file')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl',
                        help='Path to save vocabulary')

    # Optimization settings
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--max_epochs_per_trial', type=int, default=15,
                        help='Maximum epochs per trial (with early stopping)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for the study')
    parser.add_argument('--study_name', type=str, default='lstm_optimization',
                        help='Name for the Optuna study')
    parser.add_argument('--study_dir', type=str, default='./optuna_studies',
                        help='Directory to save study results')

    # Base configuration
    parser.add_argument('--max_sequence_length', type=int, default=224,
                        help='Maximum sequence length for padding')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    # WandB settings
    parser.add_argument('--project_name', type=str, default='sign-language-lstm',
                        help='WandB project name')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run WandB in offline mode')

    args = parser.parse_args()

    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    # Load or create base configuration
    if args.config:
        base_config = load_config(args.config)
    else:
        base_config = Config()
        if args.data_dir:
            base_config.data.data_dir = args.data_dir
        if args.annotations_path:
            base_config.data.annotations_path = args.annotations_path

    # Apply argument overrides
    base_config.data.vocab_path = args.vocab_path
    base_config.preprocessing.max_sequence_length = args.max_sequence_length
    base_config.device = args.device
    base_config.logging.project_name = args.project_name

    # Run optimization
    optimizer = OptunaOptimizer(base_config, args)
    study = optimizer.run_optimization(n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print("Study complete! Use the best parameters to train your final model.")
    print("=" * 60)


if __name__ == "__main__":
    main()