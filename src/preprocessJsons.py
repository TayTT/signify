"""
Enhanced Sign Language Data Preprocessor with Phoenix Dataset Support

This module provides preprocessing functionality to convert JSON landmark data
from the sign language processing pipeline into tensor-ready format for ML models,
with specific support for Phoenix dataset and LSTM training.
"""

import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import pickle
from curriculum_learning import (
    CurriculumDataset, CurriculumScheduler, CurriculumSampler,
    analyze_dataset_difficulty, SampleDifficulty
)

CORE_POSE_LANDMARKS = [
    'NOSE',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_HIP', 'RIGHT_HIP',
]

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
    max_missing_frames: int = 50 # Maximum consecutive missing frames to interpolate

    # Output format
    output_format: str = "tensor"  # "tensor", "numpy", or "dict"
    device: str = "cpu"

    # Phoenix dataset specific
    phoenix_data_path: Optional[str] = None
    phoenix_annotations_path: Optional[str] = None
    vocab_size: int = 1000  # Maximum vocabulary size for glosses


class PhoenixDataset(Dataset):
    """Dataset class for Phoenix sign language data"""

    def __init__(self,
                 json_paths: List[str],
                 annotations: List[str],
                 preprocessor: 'SignLanguagePreprocessor',
                 vocab_path: Optional[str] = None):
        """
        Initialize Phoenix dataset

        Args:
            json_paths: List of paths to JSON files
            annotations: List of annotation strings (glosses)
            preprocessor: Initialized preprocessor
            vocab_path: Path to vocabulary file (optional)
        """
        self.json_paths = json_paths
        self.annotations = annotations
        self.preprocessor = preprocessor

        # DEBUG: Test the first sample
        if len(json_paths) > 0:
            self.preprocessor.debug_single_sample(json_paths[0])  # Pass the json_path

        # Build vocabulary
        self.vocab = self._build_vocabulary(vocab_path)
        self.label_encoder = LabelEncoder()
        # self.id_to_vocab = {v: k for k, v in self.vocab.items()}

        # Create label mappings
        all_glosses = []
        for annotation in annotations:
            all_glosses.extend(annotation.split())

        unique_glosses = list(set(all_glosses))
        self.label_encoder.fit(unique_glosses)

        # Add special tokens
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.vocab['<SOS>'] = 2
        self.vocab['<EOS>'] = 3

        # Update vocab with glosses
        for i, gloss in enumerate(unique_glosses):
            if gloss not in self.vocab:
                self.vocab[gloss] = len(self.vocab)

        self.vocab_size = len(self.vocab)


    def _build_vocabulary(self, vocab_path: Optional[str] = None) -> Dict[str, int]:
        """Build vocabulary from annotations"""
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_vocabulary(self, vocab_path: str):
        """Save vocabulary to file"""
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f)

    def encode_annotation(self, annotation: str) -> List[int]:
        """Encode annotation string to token IDs"""
        tokens = annotation.split()
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return [self.vocab['<SOS>']] + encoded + [self.vocab['<EOS>']]

    def decode_annotation(self, token_ids: List[int]) -> str:
        """Decode token IDs back to annotation string"""
        id_to_vocab = {v: k for k, v in self.vocab.items()}
        tokens = []

        for token_id in token_ids:
            if token_id == 0:  # Skip padding tokens
                continue
            elif token_id == self.vocab.get('<EOS>', -1):  # Stop at end-of-sequence
                break
            elif token_id == self.vocab.get('< SOS >', -1):  # Skip start-of-sequence
                continue
            else:
                token_text = id_to_vocab.get(token_id, f'<UNK_ID_{token_id}>')
                if token_text not in ['<PAD>', '<UNK>', '< SOS >', '<EOS>']:
                    tokens.append(token_text)

        result = ' '.join(tokens)

        # Debug output
        if not result.strip():
            print(f"Debug: decode_annotation got empty result from {len(token_ids)} tokens")
            print(f"First 10 token IDs: {token_ids[:10]}")
            non_zero_tokens = [tid for tid in token_ids if tid != 0]
            print(f"Non-zero tokens: {non_zero_tokens[:10]}")

        return result

    def create_curriculum_dataset(self,
                                  total_epochs: int,
                                  warmup_epochs: int = 5) -> CurriculumDataset:
        """
        Create curriculum learning version of this dataset

        Args:
            total_epochs: Total number of training epochs
            warmup_epochs: Number of epochs for warmup phase

        Returns:
            CurriculumDataset instance
        """
        print("Creating curriculum learning dataset...")

        # Analyze difficulty for all samples
        difficulty_metrics = analyze_dataset_difficulty(
            self.json_paths,
            max_seq_length=self.preprocessor.config.max_sequence_length
        )

        # Create scheduler
        scheduler = CurriculumScheduler(
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            stages=4,
            overlap_ratio=0.3
        )

        # Create curriculum dataset
        curriculum_dataset = CurriculumDataset(
            base_dataset=self,
            difficulty_metrics=difficulty_metrics,
            scheduler=scheduler
        )

        return curriculum_dataset

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        try:
            result = self.preprocessor.process_file(self.json_paths[idx])
            sequence = result['sequence']

            # DEBUG: Check feature content
            non_zero_ratio = (sequence != 0).float().mean()
            non_zero_count = (sequence != 0).sum()
            total_elements = sequence.numel()

            if idx < 5:  # Print first 5 samples
                print(f"Sample {idx}: shape={sequence.shape}, "
                      f"non_zero={non_zero_count}/{total_elements} ({non_zero_ratio:.3f})")

                # Check individual feature types
                hands_features = sequence[:, :self.preprocessor.feature_dims['hands']]
                face_features = sequence[:, self.preprocessor.feature_dims['hands']:
                                            self.preprocessor.feature_dims['hands'] +
                                            self.preprocessor.feature_dims['face']]
                pose_features = sequence[:, -self.preprocessor.feature_dims['pose']:]

                print(f"  Hands non-zero: {(hands_features != 0).float().mean():.3f}")
                print(f"  Face non-zero: {(face_features != 0).float().mean():.3f}")
                print(f"  Pose non-zero: {(pose_features != 0).float().mean():.3f}")

            # Encode annotation
            encoded_annotation = self.encode_annotation(self.annotations[idx])

            # Pad annotation to max length
            max_annotation_length = 50  # Adjust based on your data
            if len(encoded_annotation) > max_annotation_length:
                encoded_annotation = encoded_annotation[:max_annotation_length]
            else:
                encoded_annotation.extend([self.vocab['<PAD>']] * (max_annotation_length - len(encoded_annotation)))

            # Ensure consistent tensor shapes
            sequence = result['sequence']
            attention_mask = result['attention_mask']

            # Verify shapes are correct
            expected_seq_shape = (self.preprocessor.config.max_sequence_length, self.preprocessor.feature_dims['total'])
            expected_mask_shape = (self.preprocessor.config.max_sequence_length,)

            if sequence.shape != expected_seq_shape:
                if idx % 50 == 0:  # Only warn every 50 samples
                    print(
                        f"Info: Truncating sequence at idx {idx}: {sequence.shape[0]} -> {expected_seq_shape[0]} frames")
                # Create properly shaped tensor
                fixed_sequence = torch.zeros(expected_seq_shape, dtype=sequence.dtype)
                min_seq_len = min(sequence.shape[0], expected_seq_shape[0])
                min_feat_len = min(sequence.shape[1], expected_seq_shape[1])
                fixed_sequence[:min_seq_len, :min_feat_len] = sequence[:min_seq_len, :min_feat_len]
                sequence = fixed_sequence

            if attention_mask.shape != expected_mask_shape:
                if idx % 50 == 0:
                    print(
                    f"Warning: Attention mask shape mismatch at idx {idx}: {attention_mask.shape[0]} vs expected {expected_mask_shape[0]}")
                # Create properly shaped tensor
                fixed_mask = torch.zeros(expected_mask_shape, dtype=attention_mask.dtype)
                min_len = min(attention_mask.shape[0], expected_mask_shape[0])
                fixed_mask[:min_len] = attention_mask[:min_len]
                attention_mask = fixed_mask

            return {
                'sequence': sequence,
                'attention_mask': attention_mask,
                'labels': torch.tensor(encoded_annotation, dtype=torch.long),
                'annotation': self.annotations[idx],
                'metadata': result['metadata']
            }
        except Exception as e:
            print(f"Error loading sample {idx} (file: {self.json_paths[idx]}): {e}")
            # Return a dummy sample with correct shapes
            return {
                'sequence': torch.zeros(self.preprocessor.config.max_sequence_length,
                                        self.preprocessor.feature_dims['total']),
                'attention_mask': torch.zeros(self.preprocessor.config.max_sequence_length, dtype=torch.bool),
                'labels': torch.zeros(50, dtype=torch.long),  # max_annotation_length
                'annotation': '',
                'metadata': {}
            }


class SignLanguagePreprocessor:
    """
    Enhanced preprocessor for converting JSON landmark data to ML-ready tensors
    with Phoenix dataset support
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
            pose_dim = len(CORE_POSE_LANDMARKS) * 3
            if self.config.include_pose_visibility:
                pose_dim += len(CORE_POSE_LANDMARKS)  # Visibility scores
            dims['pose'] = pose_dim

        dims['total'] = dims['hands'] + dims['face'] + dims['pose']
        return dims

    def load_phoenix_annotations(self, excel_path: str) -> pd.DataFrame:
        """
        Load Phoenix dataset annotations from Excel or CSV file

        Args:
            excel_path: Path to Excel or CSV file with annotations

        Returns:
            DataFrame with columns: id, folder, signer, annotation
        """
        try:
            file_path = Path(excel_path)
            print(f"Loading Phoenix annotations from: {file_path}")

            # Step 1: Try to load file with automatic format detection
            df = None

            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                print(" Loaded Excel file")
            else:
                # For CSV files, try different separators
                separators = ['|', ',', '\t', ';']  # Put | first since it's common for Phoenix

                print(" Trying different separators...")
                for sep in separators:
                    try:
                        test_df = pd.read_csv(file_path, sep=sep, nrows=3)
                        print(f"  Separator '{sep}': {len(test_df.columns)} columns")

                        if len(test_df.columns) >= 4:  # Need at least 4 columns
                            df = pd.read_csv(file_path, sep=sep)
                            print(f" Successfully loaded with separator '{sep}'")
                            break
                    except Exception as e:
                        print(f"  Separator '{sep}': Failed")
                        continue

                if df is None:
                    # Last resort: try reading as pipe-separated
                    try:
                        df = pd.read_csv(file_path, sep='|', engine='python')
                        print(" Fallback: loaded with pipe separator")
                    except Exception as e:
                        raise ValueError(f"Could not parse file with any separator: {e}")

            if df is None:
                raise ValueError("Failed to load annotations file")

            # Step 2: Validate and fix column names
            expected_columns = ['id', 'folder', 'signer', 'annotation']

            print(f" Initial shape: {df.shape}")
            print(f" Columns found: {list(df.columns)}")

            # Check if we have the exact column names
            if all(col in df.columns for col in expected_columns):
                print(" All expected columns found")
            else:
                print("️  Column names don't match exactly, attempting to map...")

                if len(df.columns) < 4:
                    raise ValueError(f"Insufficient columns: expected 4, found {len(df.columns)}")

                # Map first 4 columns to expected names
                column_mapping = {}
                for i, expected_col in enumerate(expected_columns):
                    if i < len(df.columns):
                        old_col = df.columns[i]
                        column_mapping[old_col] = expected_col

                print(f" Column mapping: {column_mapping}")
                df = df.rename(columns=column_mapping)

            # Step 3: Keep only expected columns and clean data
            df = df[expected_columns].copy()

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
                print(f"⚠  Removed {initial_rows - final_rows} rows with missing/empty data")

            print(f" Final shape: {df.shape}")
            print(f" Unique signers: {df['signer'].nunique()}")
            print(f" Sample annotations:")
            for i, annotation in enumerate(df['annotation'].head(3)):
                print(f"   {i+1}. {annotation}")

            return df

        except Exception as e:
            print(f" Error loading Phoenix annotations: {e}")

            # Enhanced debugging
            if Path(excel_path).exists():
                try:
                    print(" File debugging info:")
                    with open(excel_path, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline().strip() for _ in range(3)]

                    print(f"   File size: {Path(excel_path).stat().st_size} bytes")
                    print(f"   First 3 lines:")
                    for i, line in enumerate(first_lines):
                        line_preview = line[:80] + "..." if len(line) > 80 else line
                        print(f"     {i+1}: {line_preview}")

                except Exception as debug_error:
                    print(f"   Could not read file for debugging: {debug_error}")
            else:
                print(f"   File does not exist: {excel_path}")

            return pd.DataFrame()  # Return empty DataFrame instead of raising

    def create_phoenix_dataset(self,
                              data_dir: str,
                              annotations_path: str,
                              vocab_path: Optional[str] = None) -> PhoenixDataset:
        """
        Create Phoenix dataset from JSON files and annotations

        Args:
            data_dir: Directory containing JSON files
            annotations_path: Path to Excel file with annotations
            vocab_path: Path to vocabulary file (optional)

        Returns:
            PhoenixDataset instance
        """
        # Load annotations
        annotations_df = self.load_phoenix_annotations(annotations_path)

        if annotations_df.empty:
            raise ValueError("Could not load annotations")

        # Find corresponding JSON files
        json_paths = []
        valid_annotations = []

        data_path = Path(data_dir)

        for _, row in annotations_df.iterrows():
            json_file = data_path / f"{row['id']}.json"

            if json_file.exists():
                json_paths.append(str(json_file))
                valid_annotations.append(row['annotation'])
            else:
                print(f"Warning: JSON file not found for {row['id']}")

        print(f"Found {len(json_paths)} valid JSON files out of {len(annotations_df)} annotations")

        # Create dataset
        dataset = PhoenixDataset(
            json_paths=json_paths,
            annotations=valid_annotations,
            preprocessor=self,
            vocab_path=vocab_path
        )

        return dataset

    def extract_hand_features(self, hand_data: Dict) -> np.ndarray:
        """Extract hand features from hand data - with debugging"""
        features = []

        print(f"  extract_hand_features called with: {type(hand_data)}")
        print(f"  hand_data keys: {list(hand_data.keys()) if isinstance(hand_data, dict) else 'not a dict'}")

        for hand_type in ['left_hand', 'right_hand']:
            print(f"    Processing {hand_type}...")
            hand_info = hand_data.get(hand_type, {})

            if isinstance(hand_info, dict) and 'landmarks' in hand_info:
                # New format with confidence
                landmarks = hand_info['landmarks']
                confidence = hand_info.get('confidence', 1.0)
                print(f"      Found {len(landmarks)} landmarks (dict format)")
            elif isinstance(hand_info, list) and len(hand_info) > 0:
                # Old format - direct list
                landmarks = hand_info
                confidence = 1.0
                print(f"      Found {len(landmarks)} landmarks (list format)")
            else:
                # No hand detected - fill with zeros
                landmarks = []
                confidence = 0.0
                print(f"      No landmarks found for {hand_type}")

            # Extract coordinates - ensure exactly 21 landmarks
            hand_coords = []
            valid_landmarks = 0
            for i in range(self.config.hand_landmarks_count):  # Should be 21
                if i < len(landmarks):
                    lm = landmarks[i]
                    if isinstance(lm, dict) and all(k in lm for k in ['x', 'y', 'z']):
                        coords = [lm['x'], lm['y'], lm['z']]
                        hand_coords.extend(coords)
                        valid_landmarks += 1
                        if i == 0:  # Print first landmark as sample
                            print(f"      Sample landmark 0: x={lm['x']:.3f}, y={lm['y']:.3f}, z={lm['z']:.3f}")
                    else:
                        # Invalid landmark data
                        hand_coords.extend([0.0, 0.0, 0.0])
                        if i == 0:
                            print(f"      Invalid landmark at {i}: {lm}")
                else:
                    # Missing landmark
                    hand_coords.extend([0.0, 0.0, 0.0])

            features.extend(hand_coords)

            if self.config.include_hand_confidence:
                features.append(confidence)

            print(f"      {hand_type}: {valid_landmarks}/21 valid landmarks, confidence={confidence:.3f}")

        result = np.array(features, dtype=np.float32)
        print(f"  Total hand features: {len(result)}, non-zero: {(result != 0).sum()}")
        return result

    def extract_face_features(self, face_data: Dict) -> np.ndarray:
        """Extract face features from face data"""
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
        """Extract pose features from pose data"""
        features = []

        # Extract coordinates
        for landmark_name in CORE_POSE_LANDMARKS:
            if landmark_name in pose_data:
                lm = pose_data[landmark_name]
                features.extend([lm['x'], lm['y'], lm['z']])
            else:
                features.extend([0.0, 0.0, 0.0])

        # Extract visibility if requested
        if self.config.include_pose_visibility:
            for landmark_name in CORE_POSE_LANDMARKS:
                if landmark_name in pose_data:
                    visibility = pose_data[landmark_name].get('visibility', 1.0)
                    features.append(visibility)
                else:
                    features.append(0.0)

        return np.array(features, dtype=np.float32)

    def extract_frame_features(self, frame_data: Dict) -> np.ndarray:
        """Extract all features from a single frame"""
        features = []

        #---BEGIN DEBUG----
        # Debug: Print what we're starting with
        hands_data = frame_data.get('hands', {})
        face_data = frame_data.get('face', {})
        pose_data = frame_data.get('pose', {})

        print(f"\n=== Frame Feature Extraction Debug ===")
        print(f"Input frame_data keys: {list(frame_data.keys())}")
        print(
            f"Hands data type: {type(hands_data)}, keys: {list(hands_data.keys()) if isinstance(hands_data, dict) else 'not dict'}")
        print(
            f"Face data type: {type(face_data)}, keys: {list(face_data.keys()) if isinstance(face_data, dict) else 'not dict'}")
        print(
            f"Pose data type: {type(pose_data)}, keys: {list(pose_data.keys()) if isinstance(pose_data, dict) else 'not dict'}")

        # Extract hand features
        if self.config.include_hands:
            print(f"\n--- Extracting Hand Features ---")
            hand_features = self.extract_hand_features(hands_data)
            print(f"Hand features shape: {hand_features.shape}")
            print(f"Hand features non-zero ratio: {(hand_features != 0).mean():.3f}")
            print(f"Hand features range: [{hand_features.min():.3f}, {hand_features.max():.3f}]")
            features.append(hand_features)

        # Extract face features
        if self.config.include_face:
            print(f"\n--- Extracting Face Features ---")
            face_features = self.extract_face_features(face_data)
            print(f"Face features shape: {face_features.shape}")
            print(f"Face features non-zero ratio: {(face_features != 0).mean():.3f}")
            print(f"Face features range: [{face_features.min():.3f}, {face_features.max():.3f}]")
            features.append(face_features)

        # Extract pose features
        if self.config.include_pose:
            print(f"\n--- Extracting Pose Features ---")
            pose_features = self.extract_pose_features(pose_data)
            print(f"Pose features shape: {pose_features.shape}")
            print(f"Pose features non-zero ratio: {(pose_features != 0).mean():.3f}")
            print(f"Pose features range: [{pose_features.min():.3f}, {pose_features.max():.3f}]")
            features.append(pose_features)

        # Concatenate all features
        if features:
            result = np.concatenate(features)
            print(f"\n--- Final Concatenated Features ---")
            print(f"Final features shape: {result.shape}")
            print(f"Final non-zero ratio: {(result != 0).mean():.3f}")
            print(f"Final range: [{result.min():.3f}, {result.max():.3f}]")
            return result
        else:
            print(f"\n--- No Features Extracted ---")
            return np.zeros(self.feature_dims['total'], dtype=np.float32)
        #--------END DEBUFG-------
        # Extract hand features
        if self.config.include_hands:
            hand_features = self.extract_hand_features(frame_data.get('hands', {}))
            # Ensure consistent size
            expected_hand_size = self.feature_dims['hands']
            if len(hand_features) != expected_hand_size:
                print(f"Warning: Hand features size mismatch: {len(hand_features)} vs expected {expected_hand_size}")
                # Pad or truncate to expected size
                padded_features = np.zeros(expected_hand_size, dtype=np.float32)
                min_len = min(len(hand_features), expected_hand_size)
                padded_features[:min_len] = hand_features[:min_len]
                hand_features = padded_features
            features.append(hand_features)

        # Extract face features
        if self.config.include_face:
            face_features = self.extract_face_features(frame_data.get('face', {}))
            # Ensure consistent size
            expected_face_size = self.feature_dims['face']
            if len(face_features) != expected_face_size:
                print(f"Warning: Face features size mismatch: {len(face_features)} vs expected {expected_face_size}")
                # Pad or truncate to expected size
                padded_features = np.zeros(expected_face_size, dtype=np.float32)
                min_len = min(len(face_features), expected_face_size)
                padded_features[:min_len] = face_features[:min_len]
                face_features = padded_features
            features.append(face_features)

        # Extract pose features
        if self.config.include_pose:
            pose_features = self.extract_pose_features(frame_data.get('pose', {}))
            # Ensure consistent size
            expected_pose_size = self.feature_dims['pose']
            if len(pose_features) != expected_pose_size:
                print(f"Warning: Pose features size mismatch: {len(pose_features)} vs expected {expected_pose_size}")
                # Pad or truncate to expected size
                padded_features = np.zeros(expected_pose_size, dtype=np.float32)
                min_len = min(len(pose_features), expected_pose_size)
                padded_features[:min_len] = pose_features[:min_len]
                pose_features = padded_features
            features.append(pose_features)

        # Concatenate all features
        if features:
            result = np.concatenate(features)
            # Final shape check
            if len(result) != self.feature_dims['total']:
                print(f"Warning: Total features size mismatch: {len(result)} vs expected {self.feature_dims['total']}")
                # Ensure correct final size
                final_features = np.zeros(self.feature_dims['total'], dtype=np.float32)
                min_len = min(len(result), self.feature_dims['total'])
                final_features[:min_len] = result[:min_len]
                return final_features
            return result
        else:
            return np.zeros(self.feature_dims['total'], dtype=np.float32)

    def normalize_coordinates(self, features: np.ndarray) -> np.ndarray:
        """Normalize coordinates - with debugging"""
        if not self.config.normalize_coordinates:
            return features

        print(f"\n--- Coordinate Normalization ---")
        print(f"Before normalization: shape={features.shape}, range=[{features.min():.3f}, {features.max():.3f}]")
        print(f"Before normalization: non-zero ratio={((features != 0).mean()):.3f}")

        min_val, max_val = self.config.coordinate_range
        normalized = features * (max_val - min_val) + min_val

        print(f"After normalization: range=[{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"After normalization: non-zero ratio={((normalized != 0).mean()):.3f}")

        return normalized

    def interpolate_missing_frames(self, sequence: np.ndarray,
                                   valid_mask: np.ndarray) -> np.ndarray:
        """Interpolate missing frames in a sequence"""
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
        """Pad sequence to target length"""
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

    def load_json_data(self, json_path: Union[str, Path]) -> Dict:
        """Load landmarks data from JSON file"""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'frames' not in data:
            raise ValueError("Invalid JSON format: 'frames' key not found")

        return data

    def process_sequence(self, json_data: Dict) -> Tuple[np.ndarray, Dict]:
        """Process a complete sequence from JSON data"""
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
        """Process a single JSON file and return tensor-ready data"""
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
        """Process multiple JSON files in batch"""
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

    def debug_single_sample(self, json_path: str):
        """Debug processing of a single sample"""
        print(f"\n=== DEBUGGING SINGLE SAMPLE ===")
        print(f"Processing: {json_path}")

        # Load raw JSON
        json_data = self.load_json_data(json_path)
        frames_data = json_data['frames']
        first_frame_key = list(frames_data.keys())[0]
        first_frame = frames_data[first_frame_key]

        print(f"Raw frame data keys: {list(first_frame.keys())}")

        # Check hands data structure
        hands_data = first_frame.get('hands', {})
        print(f"Raw hands data type: {type(hands_data)}")

        if isinstance(hands_data, dict):
            for hand_type in ['left_hand', 'right_hand']:
                hand_info = hands_data.get(hand_type, {})
                print(f"{hand_type}: type={type(hand_info)}")
                if isinstance(hand_info, dict) and 'landmarks' in hand_info:
                    landmarks = hand_info['landmarks']
                    print(f"  {hand_type} landmarks count: {len(landmarks)}")
                    if landmarks:
                        first_landmark = landmarks[0]
                        print(f"  First landmark sample: {first_landmark}")
                elif isinstance(hand_info, list):
                    print(f"  {hand_type} landmarks count: {len(hand_info)}")
                    if hand_info:
                        print(f"  First landmark sample: {hand_info[0]}")
                else:
                    print(f"  {hand_type}: No landmarks found - {hand_info}")

        # Test individual frame extraction
        print(f"\n--- Testing single frame extraction ---")
        features = self.extract_frame_features(first_frame)
        print(f"Single frame features shape: {features.shape}")
        print(f"Single frame non-zero ratio: {(features != 0).mean():.3f}")

        # Process through full pipeline
        print(f"\n--- Processing through full pipeline ---")
        sequence, metadata = self.process_sequence(json_data)
        print(f"Processed sequence shape: {sequence.shape}")
        print(f"Processed sequence non-zero ratio: {(sequence != 0).mean():.3f}")

        return sequence




def validate_feature_dimensions(self, frame_data: Dict) -> bool:
    """Validate that extracted features match expected dimensions"""
    features = self.extract_frame_features(frame_data)
    expected_size = self.feature_dims['total']
    actual_size = len(features)

    if actual_size != expected_size:
        print(f"Feature dimension mismatch: expected {expected_size}, got {actual_size}")
        return False
    return True


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
    """Demonstration of preprocessing functionality"""
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