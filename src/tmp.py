from train_lstm import PhoenixDatasetManager
from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig

print("Loading Phoenix training data...")

dataset_manager = PhoenixDatasetManager(
    data_dir=r".\data\phoenix_jsons\phoenix_train",  # Your training data
    annotations_path=r".\data\annotations_phoenix\train_corpus.csv"
)

preprocess_config = PreprocessingConfig(
    max_sequence_length=224,
    normalize_coordinates=False,
    output_format="tensor",
    device='cpu',
    include_hand_confidence=False,
    include_pose_visibility=False
)

preprocessor = SignLanguagePreprocessor(preprocess_config)
dataset = dataset_manager.create_dataset(preprocessor)

# Check vocabulary
glosses = list(dataset.gloss_to_idx.keys())[:30]
print(f"\nVocabulary size: {dataset.vocab_size}")
print(f"First 30 glosses: {glosses}")

# Verify it's German (should NOT have DOG1, BASKETBALL1, etc.)
asl_glosses = ['DOG1', 'BASKETBALL1', 'PATIENT2']
has_asl = any(g in dataset.gloss_to_idx for g in asl_glosses)

if has_asl:
    print("\n⚠ WARNING: Found ASL glosses! Check your training data.")
else:
    print("\n✓ Looks good - only German glosses!")

    # Save the correct vocabulary
    import pickle

    vocab_data = {
        'gloss_to_idx': dataset.gloss_to_idx,
        'idx_to_gloss': dataset.idx_to_gloss
    }

    with open(r".\models_stash\phoenix-optuned\vocab_phoenix_only.pkl", 'wb') as f:
        pickle.dump(vocab_data, f)

    print(f"✓ Saved correct Phoenix vocabulary to: vocab_phoenix_only.pkl")
