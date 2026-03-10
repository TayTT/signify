"""
Unified Configuration System for Sign Language Recognition

This module provides a comprehensive configuration system that separates concerns
into distinct configuration classes and supports loading from YAML/JSON files.

Configuration Hierarchy:
    - ModelConfig: Neural network architecture settings
    - TrainingConfig: Training hyperparameters and optimization settings
    - DataConfig: Dataset paths and data processing settings
    - PreprocessingConfig: Feature extraction and landmark processing
    - LoggingConfig: WandB and experiment tracking settings
    - Config: Master configuration containing all sub-configs

Usage:
    from config import Config, load_config

    # Load from YAML file
    config = load_config("config.yaml")

    # Or create programmatically
    config = Config()

    # Access sub-configs
    model_config = config.model
    training_config = config.training
"""

import json
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from copy import deepcopy


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""

    # LSTM Architecture
    input_size: int = 356
    hidden_size: int = 768
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = False

    # Attention
    num_attention_heads: int = 1
    use_attention: bool = True

    # Sequence parameters - max_sequence_length lives in PreprocessingConfig
    max_annotation_length: int = 25

    # CTC loss option
    use_ctc: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""

    name: str = "adamw"  # adam, adamw, sgd, rmsprop
    learning_rate: float = 6.039e-5
    weight_decay: float = 3.877e-5

    # Adam/AdamW specific
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8

    # SGD specific
    momentum: float = 0.9
    nesterov: bool = False


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""

    name: str = "lambda"  # plateau, cosine, step, exponential, lambda

    # ReduceLROnPlateau
    factor: float = 0.7
    patience: int = 8
    min_lr: float = 1e-6

    # StepLR
    step_size: int = 10
    gamma: float = 0.7

    # LambdaLR (warmup + cosine decay)
    warmup_steps: int = 20
    total_steps: int = 200


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""

    # Basic training params
    batch_size: int = 4
    num_epochs: int = 50
    patience: int = 8  # Early stopping patience
    gradient_clip_norm: float = 1.0

    # Train/val split
    train_split: float = 0.8
    random_seed: int = 42

    # Gradient accumulation (for larger effective batch sizes)
    gradient_accumulation_steps: int = 1

    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Curriculum Learning
    use_curriculum_learning: bool = False
    curriculum_warmup_epochs: int = 5
    curriculum_stages: int = 4
    curriculum_overlap_ratio: float = 0.3


@dataclass
class DataConfig:
    """Dataset paths and data configuration"""

    # Paths
    data_dir: str = "./data/phoenix_jsons/phoenix_train"
    annotations_path: str = "./data/annotations_phoenix/train_corpus.csv"
    vocab_path: str = "./vocab.pkl"
    model_save_path: str = "./mode_stash/noNormNoFace/lstm_sign2gloss.pth"
    config_save_path: Optional[str] = None  # if None, saved next to model

    # Data format
    data_format: str = "auto"  # auto, npz, json


@dataclass
class PreprocessingConfig:
    """Feature extraction and preprocessing configuration"""

    # Sequence processing
    max_sequence_length: int = 224
    min_sequence_length: int = 5
    padding_strategy: str = "post"  # pre, post, center

    # Feature selection
    include_hands: bool = True
    include_face: bool = False
    include_pose: bool = True

    # Hand features
    hand_landmarks_count: int = 21
    include_hand_confidence: bool = False

    # Face features
    face_landmarks_count: int = 468
    use_face_subset: bool = True
    face_subset_indices: Optional[List[int]] = None

    # Pose features
    pose_landmarks_count: int = 25
    include_pose_visibility: bool = False

    # Normalization
    normalize_coordinates: bool = False
    coordinate_range: Tuple[float, float] = (-1.0, 1.0)

    # Data augmentation
    apply_augmentation: bool = False
    rotation_range: float = 0.1
    scale_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.01

    # Missing data handling
    interpolate_missing: bool = True
    interpolation_method: str = "linear"
    max_missing_frames: int = 50

    # Output format
    output_format: str = "tensor"
    device: str = "cpu"


@dataclass
class LoggingConfig:
    """Experiment tracking and logging configuration"""

    # WandB settings
    project_name: str = "sign-language-lstm"
    experiment_name: str = "noNormNoFace"
    tags: List[str] = field(default_factory=lambda: ["lstm", "sign-language"])
    notes: str = ""

    # WandB mode
    offline: bool = False
    disabled: bool = False

    # Logging frequency
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 1

    # Verbosity
    verbose: bool = True
    debug: bool = False


@dataclass
class ProcessingConfig:
    """MediaPipe and video processing thresholds"""

    hand_min_detection_confidence: float = 0.7
    hand_min_tracking_confidence: float = 0.5
    face_confidence_threshold: float = 0.5
    face_temporal_smoothing_frames: int = 3
    pose_min_detection_confidence: float = 0.7
    hand_min_size: float = 0.05
    hand_max_size: float = 0.5


@dataclass
class Config:
    """Master configuration containing all sub-configurations"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Runtime settings (not saved to config files)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.preprocessing.device = self.device  # sync device only

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary - tuples become lists for yaml compat"""
        def _tuples_to_lists(obj):
            if isinstance(obj, dict):
                return {k: _tuples_to_lists(v) for k, v in obj.items()}
            if isinstance(obj, (tuple, list)):
                return [_tuples_to_lists(v) for v in obj]
            return obj
        return _tuples_to_lists(asdict(self))

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save config to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        # Handle nested dataclasses
        model_dict = config_dict.get('model', {})
        training_dict = config_dict.get('training', {})
        data_dict = config_dict.get('data', {})
        preprocessing_dict = config_dict.get('preprocessing', {})
        processing_dict = config_dict.get('processing', {})
        logging_dict = config_dict.get('logging', {})

        # Handle nested optimizer and scheduler configs
        if 'optimizer' in training_dict and isinstance(training_dict['optimizer'], dict):
            training_dict['optimizer'] = OptimizerConfig(**training_dict['optimizer'])
        if 'scheduler' in training_dict and isinstance(training_dict['scheduler'], dict):
            training_dict['scheduler'] = SchedulerConfig(**training_dict['scheduler'])

        # Handle tuple conversion for preprocessing
        if 'coordinate_range' in preprocessing_dict:
            preprocessing_dict['coordinate_range'] = tuple(preprocessing_dict['coordinate_range'])
        if 'scale_range' in preprocessing_dict:
            preprocessing_dict['scale_range'] = tuple(preprocessing_dict['scale_range'])

        # backward compat: old yamls had max_sequence_length under model
        old_seq_len = model_dict.pop('max_sequence_length', None)
        if old_seq_len is not None and 'max_sequence_length' not in preprocessing_dict:
            preprocessing_dict['max_sequence_length'] = old_seq_len

        return cls(
            model=ModelConfig(**model_dict) if model_dict else ModelConfig(),
            training=TrainingConfig(**training_dict) if training_dict else TrainingConfig(),
            data=DataConfig(**data_dict) if data_dict else DataConfig(),
            preprocessing=PreprocessingConfig(**preprocessing_dict) if preprocessing_dict else PreprocessingConfig(),
            processing=ProcessingConfig(**processing_dict) if processing_dict else ProcessingConfig(),
            logging=LoggingConfig(**logging_dict) if logging_dict else LoggingConfig(),
            device=config_dict.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        )


def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from YAML or JSON file

    Args:
        path: Path to config file (.yaml, .yml, or .json)
        overrides: Optional dictionary of overrides to apply

    Returns:
        Config object

    Example:
        config = load_config("config.yaml", overrides={"training.batch_size": 8})
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load based on extension
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    # Apply overrides
    if overrides:
        config_dict = _apply_overrides(config_dict, overrides)

    return Config.from_dict(config_dict)


def _apply_overrides(config_dict: Dict, overrides: Dict[str, Any]) -> Dict:
    """Apply dotted-path overrides to config dictionary"""
    config_dict = deepcopy(config_dict)

    for key, value in overrides.items():
        parts = key.split('.')
        current = config_dict

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return config_dict


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to file (YAML or JSON based on extension)

    Args:
        config: Config object to save
        path: Output path (.yaml, .yml, or .json)
    """
    path = Path(path)

    if path.suffix in ['.yaml', '.yml']:
        config.to_yaml(path)
    elif path.suffix == '.json':
        config.to_json(path)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def create_default_config() -> Config:
    """Create a default configuration"""
    return Config()


def merge_configs(base: Config, override: Config) -> Config:
    """Merge two configs, with override taking precedence"""
    base_dict = base.to_dict()
    override_dict = override.to_dict()

    def deep_merge(base_d, override_d):
        result = deepcopy(base_d)
        for key, value in override_d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged_dict = deep_merge(base_dict, override_dict)
    return Config.from_dict(merged_dict)



def config_from_args(args) -> Config:
    """
    Create Config from argparse Namespace

    Args:
        args: argparse.Namespace with configuration arguments

    Returns:
        Config object
    """
    config = Config()

    # Map args to config
    if hasattr(args, 'data_dir'):
        config.data.data_dir = args.data_dir
    if hasattr(args, 'annotations_path'):
        config.data.annotations_path = args.annotations_path
    if hasattr(args, 'vocab_path'):
        config.data.vocab_path = args.vocab_path
    if hasattr(args, 'model_save_path'):
        config.data.model_save_path = args.model_save_path

    if hasattr(args, 'hidden_size'):
        config.model.hidden_size = args.hidden_size
    if hasattr(args, 'num_layers'):
        config.model.num_layers = args.num_layers
    if hasattr(args, 'dropout'):
        config.model.dropout = args.dropout
    if hasattr(args, 'max_sequence_length') and args.max_sequence_length:
        config.preprocessing.max_sequence_length = args.max_sequence_length
    if hasattr(args, 'max_annotation_length'):
        config.model.max_annotation_length = args.max_annotation_length

    if hasattr(args, 'batch_size'):
        config.training.batch_size = args.batch_size
    if hasattr(args, 'learning_rate'):
        config.training.optimizer.learning_rate = args.learning_rate
    if hasattr(args, 'weight_decay'):
        config.training.optimizer.weight_decay = args.weight_decay
    if hasattr(args, 'num_epochs'):
        config.training.num_epochs = args.num_epochs
    if hasattr(args, 'patience'):
        config.training.patience = args.patience
    if hasattr(args, 'gradient_clip_norm'):
        config.training.gradient_clip_norm = args.gradient_clip_norm

    if hasattr(args, 'project_name'):
        config.logging.project_name = args.project_name
    if hasattr(args, 'experiment_name'):
        config.logging.experiment_name = args.experiment_name
    if hasattr(args, 'wandb_offline') and args.wandb_offline:
        config.logging.offline = True

    if hasattr(args, 'device'):
        config.device = args.device

    return config