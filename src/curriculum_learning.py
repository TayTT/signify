#!/usr/bin/env python3
"""
Curriculum Learning Implementation for Sign Language LSTM Training

This module implements curriculum learning that progressively increases training difficulty
based on sample quality, missing data rate, and sequence length.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class SampleDifficulty:
    """Stores difficulty metrics for a single sample"""
    index: int
    sequence_length: int
    missing_data_rate: float
    quality_score: float
    overall_difficulty: float


class CurriculumScheduler:
    """Manages curriculum learning progression"""

    def __init__(self,
                 total_epochs: int,
                 warmup_epochs: int = 5,
                 stages: int = 4,
                 overlap_ratio: float = 0.3,
                 sequence_length_progression: Dict[str, int] = None):
        """
        Initialize curriculum scheduler with dynamic sequence lengths

        Args:
            total_epochs: Total number of training epochs
            warmup_epochs: Number of epochs to train only on easiest samples
            stages: Number of curriculum stages
            overlap_ratio: Overlap between stages
            sequence_length_progression: Dict mapping stage to max sequence length
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.stages = stages
        self.overlap_ratio = overlap_ratio

        # Default sequence length progression
        if sequence_length_progression is None:
            self.sequence_length_progression = {
                'warmup': 96,  # Short sequences in warmup
                'early': 128,  # Medium sequences in early stages
                'middle': 160,  # Longer sequences in middle stages
                'late': 192  # Full sequences in late stages
            }
        else:
            self.sequence_length_progression = sequence_length_progression

        # Calculate stage boundaries
        self.stage_epochs = max(1, (total_epochs - warmup_epochs) // stages)
        self.current_stage = 0

    def get_max_sequence_length(self, epoch: int) -> int:
        """Get the maximum sequence length for current epoch"""
        if epoch < self.warmup_epochs:
            return self.sequence_length_progression['warmup']

        # Calculate progression through stages
        stage_progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

        if stage_progress <= 0.25:
            return self.sequence_length_progression['early']
        elif stage_progress <= 0.6:
            return self.sequence_length_progression['middle']
        else:
            return self.sequence_length_progression['late']

    # def get_difficulty_threshold(self, epoch: int) -> float:
    #     """Get the maximum difficulty threshold for current epoch"""
    #     if epoch < self.warmup_epochs:
    #         return 0.3
    #
    #     stage_progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
    #     stage_progress = min(1.0, stage_progress)
    #
    #     return 0.3 + (0.7 * stage_progress)

    def get_difficulty_threshold(self, epoch: int) -> float:
        """Get current difficulty threshold - REVERSED to start with longer sequences"""
        progress = min(epoch / self.total_epochs, 1.0)

        # START with high difficulty (longer sequences), gradually include shorter ones
        return 1.0 - (progress * 0.7)  # Start at 1.0, go down to 0.3

    def get_sample_ratio(self, epoch: int) -> float:
        """Get the ratio of samples to include from sorted difficulty list"""
        if epoch < self.warmup_epochs:
            return 0.25

        stage_progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        stage_progress = min(1.0, stage_progress)

        return 0.25 + (0.75 * stage_progress)


class CurriculumDataset(Dataset):
    """Dataset wrapper that implements curriculum learning"""

    def __init__(self,
                 base_dataset: Dataset,
                 difficulty_metrics: List[SampleDifficulty],
                 scheduler: CurriculumScheduler,
                 preprocessor=None):
        """Initialize curriculum dataset"""
        self.base_dataset = base_dataset
        self.difficulty_metrics = difficulty_metrics
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.current_epoch = 0

        # Initialize current_sample_count immediately
        initial_ratio = self.scheduler.get_sample_ratio(0)
        self.current_sample_count = int(len(self.difficulty_metrics) * initial_ratio)
        self.current_sample_count = max(1, min(self.current_sample_count, len(self.difficulty_metrics)))

        self.use_dynamic_lengths = True

        # Sort samples by difficulty (easiest first)
        self.sorted_indices = sorted(
            range(len(difficulty_metrics)),
            key=lambda i: difficulty_metrics[i].overall_difficulty
        )

        print(f"Curriculum dataset initialized with {len(difficulty_metrics)} samples")
        print(f"Initial sample count: {self.current_sample_count}")
        print("Using fixed sequence lengths (dynamic lengths disabled)")

    def __len__(self):
        """Return current number of available samples"""
        # Always return current_sample_count, with a safe fallback
        if hasattr(self, 'current_sample_count') and self.current_sample_count > 0:
            return self.current_sample_count

        # Safe fallback
        if hasattr(self, 'difficulty_metrics') and self.difficulty_metrics:
            return max(1, len(self.difficulty_metrics) // 4)

        # Last resort fallback
        return 1

    def update_epoch(self, epoch: int):
        """Update current epoch for curriculum progression"""
        self.current_epoch = epoch

        # Get current sample ratio
        sample_ratio = self.scheduler.get_sample_ratio(epoch)
        self.current_sample_count = int(len(self.difficulty_metrics) * sample_ratio)
        self.current_sample_count = max(1, min(self.current_sample_count, len(self.difficulty_metrics)))

        print(f"Epoch {epoch}: Using {self.current_sample_count}/{len(self.difficulty_metrics)} samples "
              f"({sample_ratio:.1%}) with max difficulty {self.scheduler.get_difficulty_threshold(epoch):.3f}")

    def __getitem__(self, idx):
        """Get sample by curriculum-adjusted index"""
        # Map curriculum index to actual dataset index
        actual_idx = self.sorted_indices[idx]
        return self.base_dataset[actual_idx]

    def _resize_sample_sequence(self, sample):
        """Resize sample sequence to current curriculum length"""
        current_length = sample['sequence'].shape[0]
        target_length = self.current_max_sequence_length

        if current_length == target_length:
            return sample

        # Get original sequence and attention mask
        sequence = sample['sequence']
        attention_mask = sample['attention_mask']

        if current_length > target_length:
            # Truncate
            new_sequence = sequence[:target_length]
            new_attention_mask = attention_mask[:target_length]
        else:
            # Pad
            pad_length = target_length - current_length
            feature_dim = sequence.shape[1]

            # Create padding
            sequence_padding = torch.zeros(pad_length, feature_dim, dtype=sequence.dtype)
            mask_padding = torch.zeros(pad_length, dtype=attention_mask.dtype)

            # Concatenate
            new_sequence = torch.cat([sequence, sequence_padding], dim=0)
            new_attention_mask = torch.cat([attention_mask, mask_padding], dim=0)

        # Return updated sample
        return {
            'sequence': new_sequence,
            'attention_mask': new_attention_mask,
            'labels': sample['labels'],
            'annotation': sample['annotation'],
            'metadata': sample['metadata']
        }


def calculate_sample_difficulty(json_path: str, max_seq_length: int = 512) -> SampleDifficulty:
    """
    Calculate difficulty metrics for a single JSON file
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frames_data = data.get('frames', {})
        if not frames_data:
            return SampleDifficulty(0, 0, 1.0, 0.0, 1.0)

        # Calculate sequence length
        sequence_length = len(frames_data)

        # Calculate missing data rate and quality
        total_missing = 0
        total_components = 0
        confidence_scores = []

        for frame_key, frame_data in frames_data.items():
            # Check missing data
            missing_data = frame_data.get('missing_data', {})

            left_missing = missing_data.get('left_hand_missing', True)
            right_missing = missing_data.get('right_hand_missing', True)
            face_missing = missing_data.get('face_missing', True)

            total_missing += sum([left_missing, right_missing, face_missing])
            total_components += 3

            # Extract confidence scores from hands data
            hands_data = frame_data.get('hands', {})

            # Check left hand confidence
            left_hand_data = hands_data.get('left_hand', {})
            if isinstance(left_hand_data, dict) and 'confidence' in left_hand_data:
                confidence_scores.append(left_hand_data['confidence'])
            elif isinstance(left_hand_data, list) and left_hand_data:
                confidence_scores.append(0.8)  # Assume good quality if landmarks exist

            # Check right hand confidence
            right_hand_data = hands_data.get('right_hand', {})
            if isinstance(right_hand_data, dict) and 'confidence' in right_hand_data:
                confidence_scores.append(right_hand_data['confidence'])
            elif isinstance(right_hand_data, list) and right_hand_data:
                confidence_scores.append(0.8)  # Assume good quality if landmarks exist

            # Add face quality estimate
            face_data = frame_data.get('face', {})
            if face_data.get('all_landmarks'):
                confidence_scores.append(0.7)  # Assume reasonable face quality

        # Calculate metrics
        missing_data_rate = total_missing / max(1, total_components)

        # Calculate quality score
        if confidence_scores:
            avg_quality_score = np.mean(confidence_scores)
        else:
            # Fallback: estimate quality from missing data rate
            avg_quality_score = max(0.0, 1.0 - missing_data_rate)

        # Normalize sequence length (longer = more difficult)
        normalized_seq_length = min(1.0, sequence_length / max_seq_length)

        # Calculate overall difficulty (0 = easiest, 1 = hardest)
        difficulty_weights = {
            'missing_rate': 0.4,
            'quality': 0.4,
            'length': 0.2
        }

        overall_difficulty = (
                difficulty_weights['missing_rate'] * missing_data_rate +
                difficulty_weights['quality'] * (1.0 - avg_quality_score) +
                difficulty_weights['length'] * normalized_seq_length
        )

        return SampleDifficulty(
            index=0,  # Will be set later
            sequence_length=sequence_length,
            missing_data_rate=missing_data_rate,
            quality_score=avg_quality_score,
            overall_difficulty=overall_difficulty
        )

    except Exception as e:
        print(f"Error calculating difficulty for {json_path}: {e}")
        return SampleDifficulty(0, 0, 1.0, 0.0, 1.0)


def analyze_dataset_difficulty(json_paths: List[str],
                               max_seq_length: int = 512) -> List[SampleDifficulty]:
    """
    Analyze difficulty for entire dataset

    Args:
        json_paths: List of paths to JSON files
        max_seq_length: Maximum sequence length for normalization

    Returns:
        List of SampleDifficulty objects
    """
    print(f"Analyzing difficulty for {len(json_paths)} samples...")

    difficulties = []
    for i, json_path in enumerate(json_paths):
        difficulty = calculate_sample_difficulty(json_path, max_seq_length)
        difficulty.index = i
        difficulties.append(difficulty)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(json_paths)} samples")

    # Print statistics
    if difficulties:
        missing_rates = [d.missing_data_rate for d in difficulties]
        quality_scores = [d.quality_score for d in difficulties]
        seq_lengths = [d.sequence_length for d in difficulties]
        overall_difficulties = [d.overall_difficulty for d in difficulties]

        print(f"\nDataset Difficulty Analysis:")
        print(f"  Missing data rate: {np.mean(missing_rates):.3f} ± {np.std(missing_rates):.3f}")
        print(f"  Quality score: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}")
        print(f"  Sequence length: {np.mean(seq_lengths):.1f} ± {np.std(seq_lengths):.1f}")
        print(f"  Overall difficulty: {np.mean(overall_difficulties):.3f} ± {np.std(overall_difficulties):.3f}")

        # Show difficulty distribution
        easy_count = sum(1 for d in overall_difficulties if d < 0.3)
        medium_count = sum(1 for d in overall_difficulties if 0.3 <= d < 0.7)
        hard_count = sum(1 for d in overall_difficulties if d >= 0.7)

        print(f"  Difficulty distribution:")
        print(f"    Easy (< 0.3): {easy_count} ({easy_count / len(difficulties) * 100:.1f}%)")
        print(f"    Medium (0.3-0.7): {medium_count} ({medium_count / len(difficulties) * 100:.1f}%)")
        print(f"    Hard (>= 0.7): {hard_count} ({hard_count / len(difficulties) * 100:.1f}%)")

    return difficulties


class CurriculumSampler(Sampler):
    """Custom sampler for curriculum learning"""

    def __init__(self, curriculum_dataset: CurriculumDataset, batch_size: int):
        self.curriculum_dataset = curriculum_dataset
        self.batch_size = batch_size

    def __iter__(self):
        # Generate indices for current curriculum stage
        n = len(self.curriculum_dataset)
        indices = list(range(n))

        # Add some randomization within curriculum constraints
        # Group into batches and shuffle within each group
        batch_groups = []
        for i in range(0, len(indices), self.batch_size * 4):  # Group every 4 batches
            group = indices[i:i + self.batch_size * 4]
            np.random.shuffle(group)
            batch_groups.extend(group)

        return iter(batch_groups[:n])

    def __len__(self):
        return len(self.curriculum_dataset)