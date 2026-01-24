#!/usr/bin/env python3
"""
Inspect Model Checkpoint

This script shows what's saved in your model checkpoint file.
Usage: python inspect_checkpoint.py ./loss14/lstm_sign2gloss.pth
"""

import torch
import sys
from pathlib import Path


def inspect_checkpoint(checkpoint_path):
    """Inspect contents of a model checkpoint"""

    print(f"\n{'=' * 60}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'=' * 60}\n")

    if not Path(checkpoint_path).exists():
        print(f"Error: File not found: {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("Checkpoint Keys:")
    print("-" * 40)
    for key in checkpoint.keys():
        value = checkpoint[key]
        value_type = type(value).__name__

        if isinstance(value, dict):
            print(f"  {key:30s} : {value_type} ({len(value)} items)")
            # Show first few keys if it's a dict
            if len(value) <= 5:
                for k, v in value.items():
                    print(f"      {k}: {type(v).__name__}")
            else:
                sample_keys = list(value.keys())[:3]
                print(f"      Sample keys: {sample_keys}...")
        elif isinstance(value, (list, tuple)):
            print(f"  {key:30s} : {value_type} ({len(value)} items)")
        elif isinstance(value, (int, float, str, bool)):
            print(f"  {key:30s} : {value_type} = {value}")
        else:
            print(f"  {key:30s} : {value_type}")

    print("\n" + "=" * 60)
    print("Vocabulary Status:")
    print("-" * 40)

    has_vocab_in_checkpoint = False
    vocab_location = None

    # Check for vocabulary in checkpoint
    if 'gloss_to_idx' in checkpoint:
        has_vocab_in_checkpoint = True
        vocab_location = "checkpoint (gloss_to_idx key)"
        vocab_size = len(checkpoint['gloss_to_idx'])
        print(f"✓ Found in checkpoint: {vocab_size} glosses")
        print(f"  Sample glosses: {list(checkpoint['gloss_to_idx'].keys())[:5]}")

    # Check for vocab.pkl file
    checkpoint_dir = Path(checkpoint_path).parent
    vocab_path = checkpoint_dir / "vocab.pkl"

    if vocab_path.exists():
        import pickle
        try:
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)

            print(f"\n✓ Found vocab.pkl file")
            print(f"  Type: {type(vocab_data).__name__}")

            if isinstance(vocab_data, dict):
                print(f"  Keys: {list(vocab_data.keys())}")

                if 'gloss_to_idx' in vocab_data:
                    vocab_size = len(vocab_data['gloss_to_idx'])
                    print(f"  Vocabulary size: {vocab_size}")
                    print(f"  Sample glosses: {list(vocab_data['gloss_to_idx'].keys())[:5]}")
                elif 'vocab' in vocab_data:
                    vocab_size = len(vocab_data['vocab'])
                    print(f"  Vocabulary size: {vocab_size}")
                    print(f"  Sample glosses: {list(vocab_data['vocab'].keys())[:5]}")
                else:
                    print(f"  Vocabulary size: {len(vocab_data)}")
                    print(f"  Sample glosses: {list(vocab_data.keys())[:5]}")
        except Exception as e:
            print(f"✗ Error reading vocab.pkl: {e}")
    else:
        print(f"\n✗ vocab.pkl not found at: {vocab_path}")

    if not has_vocab_in_checkpoint and not vocab_path.exists():
        print("\n⚠ WARNING: No vocabulary found!")
        print("  The model checkpoint should contain vocabulary or have vocab.pkl nearby")

    print("\n" + "=" * 60)
    print("Model Configuration:")
    print("-" * 40)

    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"  Input size:      {getattr(config, 'input_size', 'N/A')}")
        print(f"  Hidden size:     {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  Num layers:      {getattr(config, 'num_layers', 'N/A')}")
        print(f"  Dropout:         {getattr(config, 'dropout', 'N/A')}")
        print(f"  Bidirectional:   {getattr(config, 'bidirectional', 'N/A')}")
        print(f"  Batch size:      {getattr(config, 'batch_size', 'N/A')}")
        print(f"  Learning rate:   {getattr(config, 'learning_rate', 'N/A')}")

    if 'vocab_size' in checkpoint:
        print(f"  Vocabulary size: {checkpoint['vocab_size']}")

    if 'best_val_loss' in checkpoint:
        print(f"  Best val loss:   {checkpoint['best_val_loss']:.4f}")

    if 'epoch' in checkpoint:
        print(f"  Last epoch:      {checkpoint['epoch']}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint.pth>")
        print("\nExample:")
        print("  python inspect_checkpoint.py ./loss14/lstm_sign2gloss.pth")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    inspect_checkpoint(checkpoint_path)