#!/usr/bin/env python3
"""
Diagnose Training vs Testing Mismatch

This script loads the SAME sample with both training and testing preprocessing
to find the exact difference causing the performance gap.

CONFIGURATION:
Adjust these paths in the script to match your setup:
- model_path: Path to your trained model checkpoint
- data_dir: Path to your training data
- annotations_path: Path to your training annotations
"""

import sys
import os

import torch
import numpy as np
from train_lstm import PhoenixDatasetManager
from preprocess_jsons import SignLanguagePreprocessor, PreprocessingConfig
from test_lstm import ModelTester

print("="*60)
print("DIAGNOSTIC: Training vs Testing Preprocessing")
print("="*60)

# Configure paths - adjust these to match your setup
model_path = r'C:\Users\marle\PycharmProjects\signify\models\lstm_sign2gloss.pth'

data_dir = r'C:\Users\marle\PycharmProjects\signify\data\annotations_phoenix\dev_corpus.csv'
annotations_path = r'C:\Users\marle\PycharmProjects\signify\data\phoenix_jsons_npz\phoenix_dev'
print(f"\nUsing model: {model_path}")

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please adjust the model_path variable in this script")
    sys.exit(1)

# Load model to get config
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model_config = checkpoint['config']

print(f"\nModel Config:")
print(f"  Input size: {model_config.input_size}")
print(f"  Max sequence length: {model_config.max_sequence_length}")

# TRAINING PREPROCESSING (from train_lstm.py line 496-503)
print("\n" + "-"*60)
print("TRAINING PREPROCESSING")
print("-"*60)

train_preprocess_config = PreprocessingConfig(
    max_sequence_length=model_config.max_sequence_length,
    normalize_coordinates=True,
    output_format="tensor",
    device='cpu',
    include_hand_confidence=False,
    include_pose_visibility=False
    # NOTE: Missing explicit include_hands, include_face, include_pose, use_face_subset
    # These will use DEFAULTS from PreprocessingConfig
)

train_preprocessor = SignLanguagePreprocessor(train_preprocess_config)

print(f"\nTraining preprocessor features:")
print(f"  Include hands: {train_preprocess_config.include_hands}")
print(f"  Include face: {train_preprocess_config.include_face}")
print(f"  Include pose: {train_preprocess_config.include_pose}")
print(f"  Use face subset: {train_preprocess_config.use_face_subset}")
print(f"  Normalize coords: {train_preprocess_config.normalize_coordinates}")
print(f"  Total dims: {train_preprocessor.feature_dims['total']}")

# TESTING PREPROCESSING (from test_model.py line 126-133)
print("\n" + "-"*60)
print("TESTING PREPROCESSING")
print("-"*60)

test_preprocess_config = PreprocessingConfig(
    max_sequence_length=model_config.max_sequence_length,
    normalize_coordinates=False,
    output_format="tensor",
    device='cpu',
    include_hand_confidence=False,
    include_pose_visibility=False
    # Also missing explicit include_hands, include_face, include_pose, use_face_subset
)

test_preprocessor = SignLanguagePreprocessor(test_preprocess_config)

print(f"\nTesting preprocessor features:")
print(f"  Include hands: {test_preprocess_config.include_hands}")
print(f"  Include face: {test_preprocess_config.include_face}")
print(f"  Include pose: {test_preprocess_config.include_pose}")
print(f"  Use face subset: {test_preprocess_config.use_face_subset}")
print(f"  Normalize coords: {test_preprocess_config.normalize_coordinates}")
print(f"  Total dims: {test_preprocessor.feature_dims['total']}")

# Check for differences
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

if train_preprocessor.feature_dims['total'] == test_preprocessor.feature_dims['total']:
    print("✓ Feature dimensions MATCH")
else:
    print(f"✗ Feature dimensions MISMATCH!")
    print(f"  Training: {train_preprocessor.feature_dims['total']}")
    print(f"  Testing: {test_preprocessor.feature_dims['total']}")

if train_preprocessor.feature_dims['total'] == model_config.input_size:
    print("✓ Preprocessor matches model input size")
else:
    print(f"✗ Preprocessor doesn't match model!")
    print(f"  Preprocessor: {train_preprocessor.feature_dims['total']}")
    print(f"  Model expects: {model_config.input_size}")

# Load one sample with both preprocessors
print("\n" + "="*60)
print("SAMPLE COMPARISON")
print("="*60)


print(f"\nUsing data directory: {data_dir}")
print(f"Using annotations: {annotations_path}")

if not os.path.exists(data_dir):
    print(f"ERROR: Data directory not found at {data_dir}")
    print("Please adjust the data_dir variable in this script")
    sys.exit(1)

if not os.path.exists(annotations_path):
    print(f"ERROR: Annotations file not found at {annotations_path}")
    print("Please adjust the annotations_path variable in this script")
    sys.exit(1)

dataset_manager = PhoenixDatasetManager(
    data_dir=data_dir,
    annotations_path=annotations_path
)

print("\n1. Loading with TRAINING preprocessor...")
train_dataset = dataset_manager.create_dataset(train_preprocessor)
train_sample = train_dataset[0]

print(f"\nTraining sample:")
print(f"  Sequence shape: {train_sample['sequence'].shape}")
print(f"  Min value: {train_sample['sequence'].min():.4f}")
print(f"  Max value: {train_sample['sequence'].max():.4f}")
print(f"  Mean value: {train_sample['sequence'].mean():.4f}")
print(f"  Std value: {train_sample['sequence'].std():.4f}")

print("\n2. Loading with TESTING preprocessor...")
test_dataset = dataset_manager.create_dataset(test_preprocessor)
test_sample = test_dataset[0]

print(f"\nTesting sample:")
print(f"  Sequence shape: {test_sample['sequence'].shape}")
print(f"  Min value: {test_sample['sequence'].min():.4f}")
print(f"  Max value: {test_sample['sequence'].max():.4f}")
print(f"  Mean value: {test_sample['sequence'].mean():.4f}")
print(f"  Std value: {test_sample['sequence'].std():.4f}")

# Compare
print("\n" + "-"*60)
if torch.allclose(train_sample['sequence'], test_sample['sequence'], atol=1e-6):
    print("✓✓✓ Sequences are IDENTICAL!")
    print("Preprocessing is NOT the issue.")
else:
    print("✗✗✗ Sequences are DIFFERENT!")
    print("This is the problem!")

    diff = (train_sample['sequence'] - test_sample['sequence']).abs()
    print(f"\nDifference stats:")
    print(f"  Max difference: {diff.max():.4f}")
    print(f"  Mean difference: {diff.mean():.4f}")
    print(f"  Non-zero differences: {(diff > 1e-6).sum()}/{diff.numel()}")

# Test with model
print("\n" + "="*60)
print("MODEL PREDICTION COMPARISON")
print("="*60)

tester = ModelTester(model_path, device='cpu')

print("\n1. Prediction with TRAINING preprocessed data:")
seq_train = train_sample['sequence'].unsqueeze(0)
mask_train = train_sample['attention_mask'].unsqueeze(0)

with torch.no_grad():
    out_train = tester.model(seq_train, mask_train, train_sample['labels'].unsqueeze(0))
    loss_train = out_train['loss'].item()
    pred_train = out_train['predictions'][0][:10]

print(f"  Loss: {loss_train:.4f}")
print(f"  Predictions: {pred_train}")

print("\n2. Prediction with TESTING preprocessed data:")
seq_test = test_sample['sequence'].unsqueeze(0)
mask_test = test_sample['attention_mask'].unsqueeze(0)

with torch.no_grad():
    out_test = tester.model(seq_test, mask_test, test_sample['labels'].unsqueeze(0))
    loss_test = out_test['loss'].item()
    pred_test = out_test['predictions'][0][:10]

print(f"  Loss: {loss_test:.4f}")
print(f"  Predictions: {pred_test}")

print("\n" + "-"*60)
if abs(loss_train - loss_test) < 0.01:
    print("✓ Losses are similar - preprocessing is consistent")
else:
    print(f"✗ Losses are DIFFERENT!")
    print(f"  Training preprocessing → Loss: {loss_train:.4f}")
    print(f"  Testing preprocessing → Loss: {loss_test:.4f}")
    print(f"  Difference: {abs(loss_train - loss_test):.4f}")

print("\n" + "="*60)
print("END DIAGNOSTIC")
print("="*60)