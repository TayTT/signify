#!/usr/bin/env python3
"""
Batch Prediction Module for Sign Language Recognition

This module processes multiple NPZ/JSON files and outputs predictions
along with ground truth annotations (if available).

Usage:
    python batch_predict.py --model_path ./models/lstm_sign2gloss.pth --input_dir ./data --n 5
    python batch_predict.py --model_path ./models/lstm_sign2gloss.pth --input_dir ./data --annotations_path ./annotations.csv
    python batch_predict.py --model_path ./models/lstm_sign2gloss.pth --file_list file1.npz file2.json --annotations_path ./annotations.csv
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import SignLanguageLSTM, ModelConfig
from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig


def remove_consecutive_duplicates(text: str) -> str:
    """
    Remove consecutive duplicate words from text.

    Examples:
        "MORGEN MORGEN MORGEN" -> "MORGEN"
        "MORGEN REGEN MORGEN" -> "MORGEN REGEN MORGEN"

    Args:
        text: Space-separated word sequence

    Returns:
        Text with consecutive duplicates removed
    """
    if not text:
        return text

    words = text.split()
    if not words:
        return text

    result = [words[0]]
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)

    return " ".join(result)


def count_words(text: str) -> Dict[str, int]:
    """
    Count occurrences of each word in text.

    Args:
        text: Space-separated word sequence

    Returns:
        Dictionary mapping words to their counts
    """
    if not text:
        return {}

    words = text.split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    return counts


class BatchPredictor:
    """Batch prediction handler for NPZ/JSON files"""

    def __init__(self, model_path: str, annotations_path: Optional[str] = None,
                 device: str = 'cuda', full_prediction: bool = False,
                 count_prediction: bool = False):
        """
        Initialize batch predictor

        Args:
            model_path: Path to trained model checkpoint
            annotations_path: Path to annotations file (CSV/Excel)
            device: Device to use for inference (cuda/cpu)
            full_prediction: If True, display full predictions with consecutive duplicates
            count_prediction: If True, count word occurrences in predictions and ground truth
        """
        self.model_path = Path(model_path)
        self.annotations_path = Path(annotations_path) if annotations_path else None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.annotations_df = None
        self.full_prediction = full_prediction
        self.count_prediction = count_prediction

        print(f"Loading model from: {self.model_path}")
        self.load_model()
        self.initialize_preprocessor()

        if self.annotations_path:
            self.load_annotations()

    def load_model(self):
        """Load trained model and vocabulary"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        self.vocab_size = checkpoint['vocab_size']

        self.gloss_to_idx = None
        self.idx_to_gloss = None

        if 'gloss_to_idx' in checkpoint:
            self.gloss_to_idx = checkpoint['gloss_to_idx']
            self.idx_to_gloss = checkpoint['idx_to_gloss']
        else:
            vocab_path = self.model_path.parent / "vocab.pkl"
            if vocab_path.exists():
                import pickle
                with open(vocab_path, 'rb') as f:
                    vocab_data = pickle.load(f)

                if isinstance(vocab_data, dict):
                    if 'gloss_to_idx' in vocab_data:
                        self.gloss_to_idx = vocab_data['gloss_to_idx']
                        self.idx_to_gloss = vocab_data['idx_to_gloss']
                    elif 'vocab' in vocab_data:
                        self.gloss_to_idx = vocab_data['vocab']
                        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
                    else:
                        self.gloss_to_idx = vocab_data
                        self.idx_to_gloss = {v: k for k, v in vocab_data.items()}

        if self.gloss_to_idx is None:
            print("Warning: Vocabulary not found. Predictions will show token IDs.")

        self.model = SignLanguageLSTM(self.config, self.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Device: {self.device}")

    def initialize_preprocessor(self):
        """Initialize preprocessor with model config"""
        preprocess_config = PreprocessingConfig(
            max_sequence_length=self.config.max_sequence_length,
            normalize_coordinates=True,
            output_format="tensor",
            device=str(self.device),
            include_hand_confidence=False,
            include_pose_visibility=False
        )
        self.preprocessor = SignLanguagePreprocessor(preprocess_config)

    def load_annotations(self):
        """Load annotations from CSV/Excel file"""
        if not self.annotations_path.exists():
            print(f"Warning: Annotations file not found: {self.annotations_path}")
            return

        print(f"Loading annotations from: {self.annotations_path}")

        try:
            if self.annotations_path.suffix.lower() == '.xlsx':
                self.annotations_df = pd.read_excel(self.annotations_path)
            else:
                separators = ['|', ',', '\t', ';']
                for sep in separators:
                    try:
                        test_df = pd.read_csv(self.annotations_path, sep=sep, nrows=3)
                        if len(test_df.columns) >= 2:
                            self.annotations_df = pd.read_csv(self.annotations_path, sep=sep)
                            break
                    except:
                        continue

            if self.annotations_df is None:
                print("Warning: Could not load annotations file")
                return

            print(f"Loaded {len(self.annotations_df)} annotations")
            print(f"Columns: {list(self.annotations_df.columns)}")

            required_cols = ['id', 'annotation']
            if not all(col in self.annotations_df.columns for col in required_cols):
                if len(self.annotations_df.columns) >= 2:
                    print("Mapping columns to expected format...")
                    self.annotations_df.columns = ['id', 'annotation'] + list(self.annotations_df.columns[2:])
                else:
                    print("Warning: Annotations file missing required columns")
                    self.annotations_df = None

        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.annotations_df = None

    def load_ground_truth(self, file_path: Path) -> Optional[str]:
        """
        Attempt to load ground truth annotation from various sources

        Args:
            file_path: Path to the data file

        Returns:
            Ground truth annotation string if available, None otherwise
        """
        if self.annotations_df is not None:
            file_id = file_path.stem

            matching_rows = self.annotations_df[self.annotations_df['id'] == file_id]
            if not matching_rows.empty:
                annotation = matching_rows.iloc[0]['annotation']
                if pd.notna(annotation):
                    return str(annotation).strip()

            name_parts = file_id.rsplit('-', 1)
            if len(name_parts) == 2:
                base_name = name_parts[0]
                matching_rows = self.annotations_df[self.annotations_df['id'] == base_name]
                if not matching_rows.empty:
                    annotation = matching_rows.iloc[0]['annotation']
                    if pd.notna(annotation):
                        return str(annotation).strip()

        if file_path.suffix == '.npz':
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'annotation' in data:
                    annotation = data['annotation']
                    if isinstance(annotation, np.ndarray):
                        annotation = str(annotation.item())
                    return annotation
                elif 'gloss' in data:
                    gloss = data['gloss']
                    if isinstance(gloss, np.ndarray):
                        gloss = str(gloss.item())
                    return gloss
            except Exception as e:
                pass

        elif file_path.suffix == '.json':
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'annotation' in data:
                        return data['annotation']
                    elif 'gloss' in data:
                        return data['gloss']
                    elif 'metadata' in data and 'annotation' in data['metadata']:
                        return data['metadata']['annotation']
            except Exception as e:
                pass

        txt_annotation = file_path.with_suffix('.txt')
        if txt_annotation.exists():
            try:
                with open(txt_annotation, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                pass

        return None

    def predict_single_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Make prediction for a single file

        Args:
            file_path: Path to NPZ or JSON file

        Returns:
            Dictionary containing prediction results
        """
        file_path = Path(file_path)

        try:
            if file_path.suffix == '.npz':
                result = self.preprocessor.process_file_fast(file_path)
            elif file_path.suffix == '.json':
                result = self.preprocessor.process_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            sequence = result['sequence'].unsqueeze(0).to(self.device)
            attention_mask = result['attention_mask'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(sequence, attention_mask)
                predictions = outputs['predictions'][0].cpu().numpy()

            predicted_glosses = []
            predicted_glosses_with_special = []

            for token_id in predictions:
                if self.idx_to_gloss is not None:
                    gloss = self.idx_to_gloss.get(int(token_id), f"UNK_{token_id}")
                    predicted_glosses_with_special.append(gloss)
                    if token_id > 3:
                        predicted_glosses.append(gloss)
                else:
                    predicted_glosses_with_special.append(str(token_id))
                    if token_id > 3:
                        predicted_glosses.append(str(token_id))

            prediction_full = " ".join(predicted_glosses)
            prediction_abbreviated = remove_consecutive_duplicates(prediction_full)
            prediction_with_special = " ".join(predicted_glosses_with_special)

            ground_truth = self.load_ground_truth(file_path)

            prediction_counts = count_words(prediction_with_special) if self.count_prediction else None
            ground_truth_counts = count_words(ground_truth) if self.count_prediction and ground_truth else None

            return {
                'file': str(file_path.name),
                'prediction': prediction_abbreviated,
                'prediction_full': prediction_full,
                'prediction_counts': prediction_counts,
                'ground_truth': ground_truth,
                'ground_truth_counts': ground_truth_counts,
                'success': True
            }

        except Exception as e:
            return {
                'file': str(file_path.name),
                'prediction': None,
                'prediction_full': None,
                'prediction_counts': None,
                'ground_truth': None,
                'ground_truth_counts': None,
                'success': False,
                'error': str(e)
            }

    def predict_batch(self, file_paths: List[Path]) -> List[Dict]:
        """
        Make predictions for multiple files

        Args:
            file_paths: List of paths to process

        Returns:
            List of prediction results
        """
        results = []
        total = len(file_paths)

        print(f"\nProcessing {total} files...")
        print("=" * 80)

        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{total}] Processing: {file_path.name}")

            result = self.predict_single_file(file_path)
            results.append(result)

            if result['success']:
                if self.full_prediction:
                    print(f"Prediction (abbreviated): {result['prediction']}")
                    print(f"Prediction (full):        {result['prediction_full']}")
                else:
                    print(f"Prediction:    {result['prediction']}")

                if result['ground_truth']:
                    print(f"Ground Truth:  {result['ground_truth']}")
                    match = result['prediction'].strip() == result['ground_truth'].strip()
                    print(f"Match:         {'Yes' if match else 'No'}")
                else:
                    print("Ground Truth:  Not available")

                if self.count_prediction:
                    self._print_word_counts(result)
            else:
                print(f"Error:         {result['error']}")

        return results

    def _print_word_counts(self, result: Dict):
        """Print word count statistics for a single result"""
        pred_counts = result.get('prediction_counts', {})
        gt_counts = result.get('ground_truth_counts', {})

        if not pred_counts and not gt_counts:
            return

        print("\nWord Counts:")

        all_words = set()
        if pred_counts:
            all_words.update(pred_counts.keys())
        if gt_counts:
            all_words.update(gt_counts.keys())

        all_words = sorted(all_words)

        print(f"  {'Word':<20} {'Prediction':<15} {'Ground Truth':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15}")

        for word in all_words:
            pred_count = pred_counts.get(word, 0) if pred_counts else 0
            gt_count = gt_counts.get(word, 0) if gt_counts else 0
            print(f"  {word:<20} {pred_count:<15} {gt_count:<15}")

        if pred_counts:
            total_pred = sum(pred_counts.values())
            print(f"  {'-'*20} {'-'*15} {'-'*15}")
            print(f"  {'TOTAL':<20} {total_pred:<15} {sum(gt_counts.values()) if gt_counts else 0:<15}")

    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful

        with_gt = sum(1 for r in results if r['success'] and r['ground_truth'])
        matches = sum(1 for r in results
                      if r['success'] and r['ground_truth'] and
                      r['prediction'].strip() == r['ground_truth'].strip())

        print(f"Total files:       {total}")
        print(f"Successful:        {successful}")
        print(f"Failed:            {failed}")

        if with_gt > 0:
            accuracy = (matches / with_gt) * 100
            print(f"With ground truth: {with_gt}")
            print(f"Matches:           {matches}")
            print(f"Accuracy:          {accuracy:.2f}%")

        print("=" * 80)


def find_files(input_dir: Path, n: Optional[int] = None,
               file_pattern: str = "*") -> List[Path]:
    """
    Find NPZ/JSON files in directory

    Args:
        input_dir: Directory to search
        n: Maximum number of files to return
        file_pattern: Pattern to match files

    Returns:
        List of file paths
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    files = []
    for ext in ['.npz', '.json']:
        files.extend(input_dir.glob(f"{file_pattern}{ext}"))

    files.sort()

    if n is not None and n > 0:
        files = files[:n]

    return files


def main():
    parser = argparse.ArgumentParser(
        description='Batch prediction for sign language recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_predict.py --model_path ./models/model.pth --input_dir ./data --n 5
  python batch_predict.py --model_path ./models/model.pth --input_dir ./data --annotations_path ./annotations.csv
  python batch_predict.py --model_path ./models/model.pth --file_list data/f1.npz data/f2.json --annotations_path ./annotations.csv
        """
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', type=str,
                             help='Directory containing NPZ/JSON files')
    input_group.add_argument('--file_list', type=str, nargs='+',
                             help='List of specific files to process')

    parser.add_argument('--annotations_path', type=str,
                        help='Path to annotations file (CSV/Excel)')

    parser.add_argument('--n', type=int, default=None,
                        help='Number of files to process (default: all files)')

    parser.add_argument('--pattern', type=str, default='*',
                        help='File pattern to match (default: *)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu, default: cuda)')

    parser.add_argument('--full_prediction', action='store_true',
                        help='Display both abbreviated and full predictions with consecutive duplicates')

    parser.add_argument('--count_prediction', action='store_true',
                        help='Count and display word occurrences in predictions and ground truth (includes special tokens)')

    parser.add_argument('--output', type=str,
                        help='Save results to JSON file (optional)')

    args = parser.parse_args()

    predictor = BatchPredictor(args.model_path, args.annotations_path,
                               args.device, args.full_prediction, args.count_prediction)

    if args.file_list:
        file_paths = [Path(f) for f in args.file_list]
        for fp in file_paths:
            if not fp.exists():
                print(f"Error: File not found: {fp}")
                return 1
    else:
        input_dir = Path(args.input_dir)
        file_paths = find_files(input_dir, args.n, args.pattern)

        if not file_paths:
            print(f"No NPZ/JSON files found in {input_dir}")
            return 1

    results = predictor.predict_batch(file_paths)

    predictor.print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())