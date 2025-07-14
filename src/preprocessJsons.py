"""
Sign Language Data Preprocessor

This module provides preprocessing functionality to convert JSON landmark data
from the sign language processing pipeline into tensor-ready format for ML models.

Features:
- Multi-modal data handling (hands, face, pose)
- Sequence padding and normalization
- Data augmentation capabilities
- Missing data interpolation
- Configurable feature extraction
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

CORE_POSE_LANDMARKS = [
    'NOSE',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_HIP', 'RIGHT_HIP',
]

# CORE_POSE_LANDMARKS = [
#     'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
#     'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
#     'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
#     'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
#     'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
# ]
@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing parameters"""
    # Sequence processing
    max_sequence_length: int = 512
    min_sequence_length: int = 5
    padding_strategy: str = "post"  # "pre", "post", or "center"

    # Feature selection
    include_hands: bool = True
    include_face: bool = True
    include_pose: bool = True

    # Hand features
    hand_landmarks_count: int = 21
    include_hand_confidence: bool = True

    # Face features
    face_landmarks_count: int = 468  # Full MediaPipe face mesh
    use_face_subset: bool = True  # Use subset for efficiency
    face_subset_indices: Optional[List[int]] = None

    # Pose features
    pose_landmarks_count: int = 25
    include_pose_visibility: bool = True

    # Normalization
    normalize_coordinates: bool = True
    coordinate_range: Tuple[float, float] = (-1.0, 1.0)

    # Data augmentation
    apply_augmentation: bool = False
    rotation_range: float = 0.1  # radians
    scale_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.01

    # Missing data handling
    interpolate_missing: bool = True
    interpolation_method: str = "linear"  # "linear", "cubic", "nearest"
    max_missing_frames: int = 3  # Maximum consecutive missing frames to interpolate

    # Output format
    output_format: str = "tensor"  # "tensor", "numpy", or "dict"
    device: str = "cpu"


class SignLanguagePreprocessor:
    """
    Preprocessor for converting JSON landmark data to ML-ready tensors

    This class handles the conversion of sign language landmark data from JSON format
    (as produced by your video processing pipeline) into structured tensors suitable
    for training neural networks.
    """

    def __init__(self, config: PreprocessingConfig = None):
        """
        Initialize the preprocessor

        Args:
            config: Preprocessing configuration. If None, uses default config.
        """
        self.config = config or PreprocessingConfig()

        # Set up face subset indices if needed
        if self.config.use_face_subset and self.config.face_subset_indices is None:
            self.config.face_subset_indices = self._get_default_face_subset()

        # Feature dimensions
        self.feature_dims = self._calculate_feature_dimensions()

        print(f"Initialized SignLanguagePreprocessor:")
        print(f"  - Total feature dimensions: {self.feature_dims['total']}")
        print(f"  - Hand features: {self.feature_dims['hands']}")
        print(f"  - Face features: {self.feature_dims['face']}")
        print(f"  - Pose features: {self.feature_dims['pose']}")

    def _get_default_face_subset(self) -> List[int]:
        """Get default subset of face landmarks for efficiency"""
        # Key facial landmarks (lips, eyes, eyebrows, nose outline)
        face_subset = [
            # Lip contour (outer)
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191,
            # Lip contour (inner)
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191,
            # Left eye
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # Right eye
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            # Left eyebrow
            46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305,
            # Right eyebrow
            276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276, 283,
            # Nose
            1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102,
            # Face contour
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        # Remove duplicates and sort
        return sorted(list(set(face_subset)))[:64]  # Limit to 64 key points

    def _calculate_feature_dimensions(self) -> Dict[str, int]:
        """Calculate feature dimensions based on configuration"""
        dims = {'hands': 0, 'face': 0, 'pose': 0}

        if self.config.include_hands:
            # Each hand: 21 landmarks * 3 coordinates + confidence if included
            hand_dim = self.config.hand_landmarks_count * 3
            if self.config.include_hand_confidence:
                hand_dim += 1  # Confidence score per hand
            dims['hands'] = hand_dim * 2  # Left and right hand

        if self.config.include_face:
            if self.config.use_face_subset:
                face_landmarks = len(self.config.face_subset_indices)
            else:
                face_landmarks = self.config.face_landmarks_count
            dims['face'] = face_landmarks * 3  # x, y, z coordinates

        if self.config.include_pose:
            pose_dim = self.config.pose_landmarks_count * 3
            if self.config.include_pose_visibility:
                pose_dim += self.config.pose_landmarks_count  # Visibility scores
            dims['pose'] = pose_dim

        dims['total'] = dims['hands'] + dims['face'] + dims['pose']
        return dims

    def load_json_data(self, json_path: Union[str, Path]) -> Dict:
        """
        Load landmark data from JSON file

        Args:
            json_path: Path to the video_landmarks.json file

        Returns:
            Dictionary containing metadata and frame data
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'frames' not in data:
            raise ValueError("Invalid JSON format: 'frames' key not found")

        return data

    def extract_hand_features(self, hand_data: Dict) -> np.ndarray:
        """
        Extract hand features from hand data

        Args:
            hand_data: Hand landmarks data for both hands

        Returns:
            Array of shape (hand_features,) containing hand coordinates and confidence
        """
        features = []

        for hand_type in ['left_hand', 'right_hand']:
            hand_info = hand_data.get(hand_type, {})

            if isinstance(hand_info, dict) and 'landmarks' in hand_info:
                # New format with confidence
                landmarks = hand_info['landmarks']
                confidence = hand_info.get('confidence', 1.0)
            elif isinstance(hand_info, list) and len(hand_info) > 0:
                # Old format - direct list
                landmarks = hand_info
                confidence = 1.0
            else:
                # No hand detected - fill with zeros
                landmarks = []
                confidence = 0.0

            # Extract coordinates
            hand_coords = []
            if landmarks and len(landmarks) >= self.config.hand_landmarks_count:
                for i in range(self.config.hand_landmarks_count):
                    lm = landmarks[i]
                    hand_coords.extend([lm['x'], lm['y'], lm['z']])
            else:
                # Missing hand - fill with zeros
                hand_coords = [0.0] * (self.config.hand_landmarks_count * 3)

            features.extend(hand_coords)

            if self.config.include_hand_confidence:
                features.append(confidence)

        return np.array(features, dtype=np.float32)

    def extract_face_features(self, face_data: Dict) -> np.ndarray:
        """
        Extract face features from face data

        Args:
            face_data: Face landmarks data

        Returns:
            Array of shape (face_features,) containing face coordinates
        """
        features = []

        face_landmarks = face_data.get('all_landmarks', [])

        if face_landmarks:
            if self.config.use_face_subset:
                # Use subset of face landmarks
                for idx in self.config.face_subset_indices:
                    if idx < len(face_landmarks):
                        lm = face_landmarks[idx]
                        features.extend([lm['x'], lm['y'], lm['z']])
                    else:
                        features.extend([0.0, 0.0, 0.0])
            else:
                # Use all face landmarks
                for lm in face_landmarks:
                    features.extend([lm['x'], lm['y'], lm['z']])
        else:
            # No face detected - fill with zeros
            expected_landmarks = (len(self.config.face_subset_indices)
                                  if self.config.use_face_subset
                                  else self.config.face_landmarks_count)
            features = [0.0] * (expected_landmarks * 3)

        return np.array(features, dtype=np.float32)

    def extract_pose_features(self, pose_data: Dict) -> np.ndarray:
        """
        Extract pose features from pose data

        Args:
            pose_data: Pose landmarks data

        Returns:
            Array of shape (pose_features,) containing pose coordinates and visibility
        """
        features = []

        # MediaPipe pose landmark names in order
        pose_landmarks = CORE_POSE_LANDMARKS

        # Extract coordinates
        for landmark_name in pose_landmarks:
            if landmark_name in pose_data:
                lm = pose_data[landmark_name]
                features.extend([lm['x'], lm['y'], lm['z']])
            else:
                features.extend([0.0, 0.0, 0.0])

        # Extract visibility if requested
        if self.config.include_pose_visibility:
            for landmark_name in pose_landmarks:
                if landmark_name in pose_data:
                    visibility = pose_data[landmark_name].get('visibility', 1.0)
                    features.append(visibility)
                else:
                    features.append(0.0)

        return np.array(features, dtype=np.float32)

    def extract_frame_features(self, frame_data: Dict) -> np.ndarray:
        """
        Extract all features from a single frame

        Args:
            frame_data: Complete frame data including hands, face, and pose

        Returns:
            Array of shape (total_features,) containing all extracted features
        """
        features = []

        # Extract hand features
        if self.config.include_hands:
            hand_features = self.extract_hand_features(frame_data.get('hands', {}))
            features.append(hand_features)

        # Extract face features
        if self.config.include_face:
            face_features = self.extract_face_features(frame_data.get('face', {}))
            features.append(face_features)

        # Extract pose features
        if self.config.include_pose:
            pose_features = self.extract_pose_features(frame_data.get('pose', {}))
            features.append(pose_features)

        # Concatenate all features
        if features:
            return np.concatenate(features)
        else:
            return np.array([], dtype=np.float32)

    def normalize_coordinates(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize coordinate features to specified range

        Args:
            features: Feature array with coordinate data

        Returns:
            Normalized feature array
        """
        if not self.config.normalize_coordinates:
            return features

        # Assume coordinates are in [0, 1] range (MediaPipe normalized coordinates)
        # Reshape to target range
        min_val, max_val = self.config.coordinate_range

        # Normalize from [0, 1] to [min_val, max_val]
        normalized = features * (max_val - min_val) + min_val

        return normalized

    def interpolate_missing_frames(self, sequence: np.ndarray,
                                   valid_mask: np.ndarray) -> np.ndarray:
        """
        Interpolate missing frames in a sequence

        Args:
            sequence: Array of shape (seq_len, features)
            valid_mask: Boolean mask indicating valid frames

        Returns:
            Interpolated sequence
        """
        if not self.config.interpolate_missing or np.all(valid_mask):
            return sequence

        # Find missing segments
        missing_indices = np.where(~valid_mask)[0]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            warnings.warn("Not enough valid frames for interpolation")
            return sequence

        interpolated = sequence.copy()

        for missing_idx in missing_indices:
            # Find nearest valid frames
            left_valid = valid_indices[valid_indices < missing_idx]
            right_valid = valid_indices[valid_indices > missing_idx]

            if len(left_valid) > 0 and len(right_valid) > 0:
                left_idx = left_valid[-1]
                right_idx = right_valid[0]

                # Check if gap is not too large
                if right_idx - left_idx <= self.config.max_missing_frames:
                    # Linear interpolation
                    alpha = (missing_idx - left_idx) / (right_idx - left_idx)
                    interpolated[missing_idx] = (
                            (1 - alpha) * sequence[left_idx] +
                            alpha * sequence[right_idx]
                    )

        return interpolated

    def pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad sequence to target length

        Args:
            sequence: Array of shape (seq_len, features)

        Returns:
            Padded sequence of shape (max_seq_len, features)
        """
        seq_len, features = sequence.shape
        target_len = self.config.max_sequence_length

        if seq_len >= target_len:
            # Truncate if too long
            return sequence[:target_len]

        # Pad if too short
        padding_needed = target_len - seq_len

        if self.config.padding_strategy == "post":
            padding = np.zeros((padding_needed, features), dtype=sequence.dtype)
            return np.vstack([sequence, padding])
        elif self.config.padding_strategy == "pre":
            padding = np.zeros((padding_needed, features), dtype=sequence.dtype)
            return np.vstack([padding, sequence])
        elif self.config.padding_strategy == "center":
            pad_before = padding_needed // 2
            pad_after = padding_needed - pad_before
            padding_before = np.zeros((pad_before, features), dtype=sequence.dtype)
            padding_after = np.zeros((pad_after, features), dtype=sequence.dtype)
            return np.vstack([padding_before, sequence, padding_after])

        raise ValueError(f"Unknown padding strategy: {self.config.padding_strategy}")



    def process_sequence(self, json_data: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Process a complete sequence from JSON data

        Args:
            json_data: Complete JSON data with metadata and frames

        Returns:
            Tuple of (processed_sequence, metadata)
        """
        frames_data = json_data['frames']
        metadata = json_data.get('metadata', {})

        # Sort frame keys by frame number
        frame_keys = sorted(frames_data.keys(), key=int)

        if len(frame_keys) < self.config.min_sequence_length:
            raise ValueError(f"Sequence too short: {len(frame_keys)} < {self.config.min_sequence_length}")

        # Extract features for each frame
        sequence_features = []
        valid_frames = []

        for frame_key in frame_keys:
            frame_data = frames_data[frame_key]
            features = self.extract_frame_features(frame_data)

            # Check if frame has valid data
            is_valid = np.any(features != 0)
            valid_frames.append(is_valid)

            # Normalize coordinates
            features = self.normalize_coordinates(features)

            sequence_features.append(features)

        # Convert to numpy array
        sequence = np.array(sequence_features, dtype=np.float32)
        valid_mask = np.array(valid_frames, dtype=bool)

        # Interpolate missing frames
        sequence = self.interpolate_missing_frames(sequence, valid_mask)

        # Pad sequence
        sequence = self.pad_sequence(sequence)

        # Create attention mask for padding
        attention_mask = np.ones(len(frame_keys), dtype=bool)
        if len(frame_keys) < self.config.max_sequence_length:
            padding_length = self.config.max_sequence_length - len(frame_keys)
            if self.config.padding_strategy == "post":
                attention_mask = np.concatenate([attention_mask, np.zeros(padding_length, dtype=bool)])
            elif self.config.padding_strategy == "pre":
                attention_mask = np.concatenate([np.zeros(padding_length, dtype=bool), attention_mask])
            elif self.config.padding_strategy == "center":
                pad_before = padding_length // 2
                pad_after = padding_length - pad_before
                attention_mask = np.concatenate([
                    np.zeros(pad_before, dtype=bool),
                    attention_mask,
                    np.zeros(pad_after, dtype=bool)
                ])

        # Add metadata
        processing_metadata = {
            'original_length': len(frame_keys),
            'padded_length': self.config.max_sequence_length,
            'feature_dimensions': self.feature_dims,
            'valid_frames': np.sum(valid_frames),
            'attention_mask': attention_mask,
            'original_metadata': metadata
        }

        return sequence, processing_metadata

    def process_file(self, json_path: Union[str, Path]) -> Union[torch.Tensor, np.ndarray, Dict]:
        """
        Process a single JSON file and return tensor-ready data

        Args:
            json_path: Path to the video_landmarks.json file

        Returns:
            Processed data in the format specified by config.output_format
        """
        # Load and process data
        json_data = self.load_json_data(json_path)
        sequence, metadata = self.process_sequence(json_data)

        # Convert to requested output format
        if self.config.output_format == "tensor":
            sequence_tensor = torch.from_numpy(sequence).to(self.config.device)
            attention_mask = torch.from_numpy(metadata['attention_mask']).to(self.config.device)

            return {
                'sequence': sequence_tensor,
                'attention_mask': attention_mask,
                'metadata': metadata
            }

        elif self.config.output_format == "numpy":
            return {
                'sequence': sequence,
                'attention_mask': metadata['attention_mask'],
                'metadata': metadata
            }

        elif self.config.output_format == "dict":
            return {
                'sequence': sequence,
                'attention_mask': metadata['attention_mask'],
                'metadata': metadata
            }

        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")

    def process_batch(self, json_paths: List[Union[str, Path]]) -> Dict:
        """
        Process multiple JSON files in batch

        Args:
            json_paths: List of paths to video_landmarks.json files

        Returns:
            Dictionary containing batched data
        """
        sequences = []
        attention_masks = []
        metadata_list = []

        for json_path in json_paths:
            try:
                result = self.process_file(json_path)
                sequences.append(result['sequence'])
                attention_masks.append(result['attention_mask'])
                metadata_list.append(result['metadata'])
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                continue

        if not sequences:
            raise ValueError("No valid sequences were processed")

        # Stack sequences
        if self.config.output_format == "tensor":
            batched_sequences = torch.stack(sequences)
            batched_masks = torch.stack(attention_masks)
        else:
            batched_sequences = np.stack(sequences)
            batched_masks = np.stack(attention_masks)

        return {
            'sequences': batched_sequences,
            'attention_masks': batched_masks,
            'metadata': metadata_list
        }


# Example usage and utility functions
def create_default_config(**kwargs) -> PreprocessingConfig:
    """Create a default configuration with optional overrides"""
    config = PreprocessingConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    return config


def preprocess_single(json_path: str):
    """
    Demonstration of preprocessing functionality

    Args:
        json_path: Path to a video_landmarks.json file
    """
    print("=== Sign Language Data Preprocessing ===\n")

    # Create configuration
    config = create_default_config(
        max_sequence_length=256,
        normalize_coordinates=True,
        include_hand_confidence=True,
        apply_augmentation=False,
        output_format="tensor"
    )

    # Initialize preprocessor
    preprocessor = SignLanguagePreprocessor(config)

    # Process file
    try:
        result = preprocessor.process_file(json_path)

        print(f"Processed sequence shape: {result['sequence'].shape}")
        print(f"Attention mask shape: {result['attention_mask'].shape}")
        print(f"Original sequence length: {result['metadata']['original_length']}")
        print(f"Valid frames: {result['metadata']['valid_frames']}")
        print(f"Feature dimensions: {result['metadata']['feature_dimensions']}")

        return result

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    json_file = "../output/images_sample/video_landmarks.json"
    preprocess_single(json_file)