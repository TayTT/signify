#!/usr/bin/env python3
"""
Example Usage Script for Real-Time LSTM Sign Language Model Integration

This script demonstrates how to:
1. Process Phoenix dataset
2. Train real-time LSTM model
3. Perform real-time inference
4. Prepare for mBART integration

Usage:
    python example_usage.py
"""

import os
import sys
import pandas as pd
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig
from lstm_model import SignLanguageLSTM, SignLanguageTrainer, ModelConfig
from train_lstm import PhoenixDatasetManager


def create_sample_annotations():
    """Create sample annotations file for testing"""
    sample_data = {
        'id': [
            '01April_2010_Thursday_heute_default-5',
            '01April_2010_Thursday_tagesschau_default-7',
            '01April_2010_Thursday_tagesschau_default-8'
        ],
        'folder': [
            '01April_2010_Thursday_heute_default-5/1/*.png',
            '01April_2010_Thursday_tagesschau_default-7/1/*.png',
            '01April_2010_Thursday_tagesschau_default-8/1/*.png'
        ],
        'signer': ['Signer04', 'Signer04', 'Signer04'],
        'annotation': [
            'ABER FREUEN MORGEN SONNE SELTEN REGEN',
            'SAMSTAG WECHSELHAFT BESONDERS FREUNDLICH NORDOST BISSCHEN BEREICH',
            'SONNTAG REGEN TEIL GEWITTER SUEDOST DURCH REGEN'
        ]
    }

    df = pd.DataFrame(sample_data)
    df.to_excel('./phoenix_annotations.xlsx', index=False)
    print("Created sample annotations file: phoenix_annotations.xlsx")
    return df


def demo_preprocessing():
    """Demonstrate preprocessing functionality"""
    print("=== Preprocessing Demo ===")

    # Create configuration
    config = PreprocessingConfig(
        max_sequence_length=128,
        include_hands=True,
        include_face=True,
        include_pose=True,
        output_format="tensor"
    )

    # Initialize preprocessor
    preprocessor = SignLanguagePreprocessor(config)

    print(f"Preprocessor initialized with {preprocessor.feature_dims['total']} features")
    print(f"  - Hands: {preprocessor.feature_dims['hands']}")
    print(f"  - Face: {preprocessor.feature_dims['face']}")
    print(f"  - Pose: {preprocessor.feature_dims['pose']}")

    return preprocessor


def demo_dataset_creation(preprocessor):
    """Demonstrate dataset creation"""
    print("\n=== Dataset Creation Demo ===")

    # Create sample annotations if they don't exist
    if not Path('./phoenix_annotations.xlsx').exists():
        create_sample_annotations()

    # Initialize dataset manager
    dataset_manager = PhoenixDatasetManager(
        data_dir='./output',
        annotations_path='./phoenix_annotations.xlsx'
    )

    try:
        # Create dataset
        dataset = dataset_manager.create_dataset(preprocessor)

        print(f"Dataset created with {len(dataset)} samples")
        print(f"Vocabulary size: {dataset.vocab_size}")

        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample 0:")
            print(f"  Sequence shape: {sample['sequence'].shape}")
            print(f"  Attention mask shape: {sample['attention_mask'].shape}")
            print(f"  Labels shape: {sample['labels'].shape}")
            print(f"  Annotation: {sample['annotation']}")

        return dataset

    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Make sure you have processed JSON files in the output directory")
        return None


def demo_model_training(dataset):
    """Demonstrate model training for real-time processing"""
    print("\n=== Real-Time Model Training Demo ===")

    if dataset is None:
        print("Skipping training demo - no dataset available")
        return

    # Create model configuration for real-time processing
    config = ModelConfig(
        input_size=192,  # Will be updated automatically
        hidden_size=128,  # Smaller for real-time demo
        num_layers=1,  # Single layer for speed
        dropout=0.2,
        bidirectional=False,  # Real-time processing
        batch_size=4,  # Small batch for demo
        learning_rate=1e-3,
        num_epochs=2,  # Just 2 epochs for demo
        patience=5,
        max_sequence_length=64,  # Shorter sequences for real-time
        data_dir='./output',
        annotations_path='./phoenix_annotations.xlsx',
        project_name='real-time-sign-language-demo',
        experiment_name='demo-run'
    )

    # Initialize trainer
    trainer = SignLanguageTrainer(config)
    trainer.dataset = dataset
    trainer.train_loader, trainer.val_loader = trainer._create_data_loaders()

    print(f"Real-time model initialized with {dataset.vocab_size} vocabulary size")
    print(f"Model: {config.hidden_size}H x {config.num_layers}L unidirectional LSTM")
    print(f"Train samples: {len(trainer.train_loader.dataset)}")
    print(f"Validation samples: {len(trainer.val_loader.dataset)}")

    # Run training for demo (just 1 epoch)
    print("\nRunning real-time model training demo...")
    try:
        trainer.train()
        print("Training completed successfully!")

        # Evaluate sample
        trainer.evaluate_sample(0)

    except Exception as e:
        print(f"Training error: {e}")
        print("This is expected if you don't have sufficient training data")


def demo_inference():
    """Demonstrate model inference"""
    print("\n=== Inference Demo ===")

    model_path = './models/lstm_sign2gloss.pth'

    if not Path(model_path).exists():
        print("No trained model found. Run training first.")
        return

    # Load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        vocab_size = checkpoint['vocab_size']

        # Initialize model
        model = SignLanguageLSTM(config, vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Model loaded successfully")
        print(f"Architecture: {config.hidden_size}H x {config.num_layers}L unidirectional LSTM")
        print(f"Vocabulary size: {vocab_size}")

        # Demo real-time prediction
        print("\nDemo: Real-time frame processing")

        # Create dummy frame features
        dummy_features = torch.randn(1, config.input_size)

        # Initialize hidden state
        hidden_state = None

        # Process several frames
        for i in range(5):
            result = model.predict_realtime(dummy_features, hidden_state)
            hidden_state = result['hidden_state']

            print(f"Frame {i + 1}: Prediction ID={result['prediction']}, Confidence={result['confidence']:.3f}")

        print("Real-time processing demo completed!")

    except Exception as e:
        print(f"Error loading model: {e}")


def demo_realtime_inference():
    """Demonstrate real-time inference capabilities"""
    print("\n=== Real-Time Inference Demo ===")

    model_path = './models/lstm_sign2gloss.pth'
    vocab_path = './vocab.pkl'

    if not (Path(model_path).exists() and Path(vocab_path).exists()):
        print("Missing model or vocabulary files. Train the model first.")
        print("Expected files:")
        print(f"  - {model_path}")
        print(f"  - {vocab_path}")
        return

    try:
        # This would normally import the real-time recognizer
        print("Real-time inference capabilities:")
        print("1. Frame-by-frame processing")
        print("2. Hidden state persistence")
        print("3. Confidence filtering")
        print("4. Prediction smoothing")
        print("5. Performance monitoring")

        print("\nTo run real-time inference:")
        print("python realtime_inference.py \\")
        print("    --model_path ./models/lstm_sign2gloss.pth \\")
        print("    --vocab_path ./vocab.pkl \\")
        print("    --camera_id 0")

        print("\nReal-time controls:")
        print("  'r' - Reset recognition state")
        print("  'q' - Quit")
        print("  'h' - Toggle help")

    except Exception as e:
        print(f"Error in real-time demo: {e}")


def prepare_for_mbart():
    """Demonstrate preparing outputs for mBART integration"""
    print("\n=== Real-Time mBART Integration Preparation ===")

    print("Real-time pipeline for mBART integration:")
    print("1. Live video frames → LSTM model → Gloss sequences")
    print("2. Buffer gloss sequences for complete phrases")
    print("3. Feed complete phrases to mBART for translation")
    print("4. Display translated text in real-time")

    # Example output format
    example_glosses = [
        "ABER FREUEN MORGEN SONNE SELTEN REGEN",
        "SAMSTAG WECHSELHAFT BESONDERS FREUNDLICH NORDOST BISSCHEN BEREICH",
        "SONNTAG REGEN TEIL GEWITTER SUEDOST DURCH REGEN"
    ]

    print("\nExample real-time gloss outputs:")
    for i, gloss in enumerate(example_glosses):
        print(f"  Frame sequence {i + 1}: {gloss}")

    print("\nReal-time integration considerations:")
    print("1. **Phrase Detection**: Detect complete sign phrases")
    print("2. **Buffering Strategy**: Accumulate glosses until phrase completion")
    print("3. **Translation Latency**: Balance accuracy vs. response time")
    print("4. **Error Handling**: Graceful degradation for partial phrases")

    print("\nNext steps for real-time mBART integration:")
    print("1. Implement phrase boundary detection")
    print("2. Create gloss-to-text buffer management")
    print("3. Integrate mBART for Gloss2Text translation")
    print("4. Build end-to-end real-time pipeline:")
    print("   Live Video → LSTM → Phrase Buffer → mBART → Display Text")


def main():
    """Main demonstration function"""
    print("Real-Time LSTM Sign Language Model Integration Demo")
    print("=" * 55)

    # Check dependencies
    try:
        import torch
        import transformers
        import wandb
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"WandB available: {wandb.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return

    # Run demonstrations
    preprocessor = demo_preprocessing()
    dataset = demo_dataset_creation(preprocessor)
    demo_model_training(dataset)
    demo_inference()
    demo_realtime_inference()
    prepare_for_mbart()

    print("\n" + "=" * 55)
    print("Demo completed!")
    print("\nTo start training:")
    print("python train_lstm.py --data_dir ./output --annotations_path ./phoenix_annotations.xlsx")
    print("\nTo start real-time inference:")
    print("python realtime_inference.py --model_path ./models/lstm_sign2gloss.pth --camera_id 0")


if __name__ == "__main__":
    main()