#!/usr/bin/env python3
"""
Convert JSON landmark files to compressed NPZ format

This script converts existing JSON landmark files to NPZ format,
reducing file size by a lot while maintaining full data fidelity.

Usage:
    python convert_json_to_npz.py --input-dir ./output
    python convert_json_to_npz.py --input-dir ./output --recursive
    python convert_json_to_npz.py --input-dir ./output --delete-json
    python convert_json_to_npz.py --single-file ./output/video.json
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Core pose landmarks (must match your processing.py)
CORE_POSE_LANDMARKS = [
    'NOSE',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_HIP', 'RIGHT_HIP',
]


def convert_frames_to_numpy(all_frames_data: Dict, frame_keys: List) -> Dict:
    """Convert frame dict data to numpy arrays for NPZ storage"""
    num_frames = len(frame_keys)

    # Pre-allocate arrays
    hands_array = np.zeros((num_frames, 2, 21, 3), dtype=np.float32)
    face_array = np.zeros((num_frames, 468, 3), dtype=np.float32)
    pose_array = np.zeros((num_frames, len(CORE_POSE_LANDMARKS), 3), dtype=np.float32)

    # Optional metadata arrays
    hand_confidence = np.zeros((num_frames, 2), dtype=np.float32)
    hand_tracked = np.zeros((num_frames, 2), dtype=bool)
    face_detected = np.zeros(num_frames, dtype=bool)

    # Fill arrays from frame data
    for i, frame_key in enumerate(frame_keys):
        frame = all_frames_data[frame_key]

        # Extract hand landmarks
        if 'hands' in frame:
            for hand_idx, hand_type in enumerate(['left_hand', 'right_hand']):
                hand_data = frame['hands'].get(hand_type)
                if hand_data and 'landmarks' in hand_data:
                    landmarks = hand_data['landmarks']
                    for j, lm in enumerate(landmarks):
                        hands_array[i, hand_idx, j] = [lm['x'], lm['y'], lm['z']]
                    if 'confidence' in hand_data:
                        hand_confidence[i, hand_idx] = hand_data['confidence']
                    hand_tracked[i, hand_idx] = True

        # Extract face landmarks
        if 'face' in frame and 'all_landmarks' in frame['face']:
            face_landmarks = frame['face']['all_landmarks']
            for j, lm in enumerate(face_landmarks[:468]):  # Limit to 468
                face_array[i, j] = [lm['x'], lm['y'], lm['z']]
            face_detected[i] = True

        # Extract pose landmarks
        if 'pose' in frame:
            pose_data = frame['pose']
            for j, landmark_name in enumerate(CORE_POSE_LANDMARKS):
                if landmark_name in pose_data:
                    lm = pose_data[landmark_name]
                    pose_array[i, j] = [lm['x'], lm['y'], lm['z']]

    return {
        'hands': hands_array,
        'face': face_array,
        'pose': pose_array,
        'hand_confidence': hand_confidence,
        'hand_tracked': hand_tracked,
        'face_detected': face_detected,
        'frame_numbers': np.array([int(k) for k in frame_keys], dtype=np.int32)
    }


def convert_json_to_npz(json_path: Path, output_path: Path = None, verbose: bool = True) -> bool:
    """
    Convert a single JSON file to NPZ format

    Args:
        json_path: Path to input JSON file
        output_path: Path for output NPZ file (default: same as JSON with .npz extension)
        verbose: Print conversion details

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Load JSON data
        if verbose:
            print(f"Loading {json_path.name}...", end=" ")

        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'frames' not in data:
            print(f"ERROR: Invalid JSON format in {json_path}")
            return False

        all_frames_data = data['frames']
        metadata = data.get('metadata', {})

        # Convert frames to numpy
        frame_keys = sorted(all_frames_data.keys(), key=int)
        numpy_data = convert_frames_to_numpy(all_frames_data, frame_keys)

        # Determine output path
        if output_path is None:
            output_path = json_path.with_suffix('.npz')

        # Save as compressed NPZ
        np.savez_compressed(
            output_path,
            hands=numpy_data['hands'],
            face=numpy_data['face'],
            pose=numpy_data['pose'],
            hand_confidence=numpy_data['hand_confidence'],
            hand_tracked=numpy_data['hand_tracked'],
            face_detected=numpy_data['face_detected'],
            frame_numbers=numpy_data['frame_numbers'],
            # Metadata as arrays
            metadata_fps=np.array([metadata.get('fps', 25.0)], dtype=np.float32),
            metadata_total_frames=np.array([metadata.get('total_frames', len(frame_keys))], dtype=np.int32),
            metadata_width=np.array([metadata.get('width', 0)], dtype=np.int32),
            metadata_height=np.array([metadata.get('height', 0)], dtype=np.int32),
            metadata_input_source=np.array([str(metadata.get('input_source', ''))], dtype=object)
        )

        # Calculate size reduction
        json_size = json_path.stat().st_size
        npz_size = output_path.stat().st_size
        reduction = (1 - npz_size / json_size) * 100

        if verbose:
            print(
                f"✓ Converted ({json_size / 1024 / 1024:.2f} MB → {npz_size / 1024 / 1024:.2f} MB, {reduction:.1f}% reduction)")

        return True

    except Exception as e:
        print(f"ERROR converting {json_path}: {e}")
        return False


def validate_conversion(json_path: Path, npz_path: Path) -> bool:
    """
    Validate that NPZ file correctly converted from JSON

    Args:
        json_path: Original JSON file
        npz_path: Converted NPZ file

    Returns:
        True if validation passes
    """
    try:
        # Load both files
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        npz_data = np.load(npz_path, allow_pickle=True)

        # Check frame count
        json_frame_count = len(json_data['frames'])
        npz_frame_count = npz_data['hands'].shape[0]

        if json_frame_count != npz_frame_count:
            print(f"  WARNING: Frame count mismatch: JSON={json_frame_count}, NPZ={npz_frame_count}")
            return False

        # Spot check a few frames
        frame_keys = sorted(json_data['frames'].keys(), key=int)
        check_frames = [0, len(frame_keys) // 2, -1]  # First, middle, last

        for frame_idx in check_frames:
            frame_key = frame_keys[frame_idx]
            json_frame = json_data['frames'][frame_key]

            # Check left hand first landmark
            if 'hands' in json_frame and json_frame['hands'].get('left_hand'):
                json_lm = json_frame['hands']['left_hand']['landmarks'][0]
                npz_lm = npz_data['hands'][frame_idx, 0, 0]

                if not np.allclose([json_lm['x'], json_lm['y'], json_lm['z']], npz_lm, atol=1e-6):
                    print(f"  WARNING: Data mismatch at frame {frame_key}")
                    return False

        return True

    except Exception as e:
        print(f"  ERROR during validation: {e}")
        return False


def convert_directory(input_dir: Path, recursive: bool = False, delete_json: bool = False,
                      validate: bool = True, verbose: bool = True) -> Dict:
    """
    Convert all JSON files in a directory to NPZ format

    Args:
        input_dir: Directory containing JSON files
        recursive: Search subdirectories recursively
        delete_json: Delete original JSON files after successful conversion
        validate: Validate each conversion
        verbose: Print detailed progress

    Returns:
        Dictionary with conversion statistics
    """
    # Find all JSON files
    if recursive:
        json_files = list(input_dir.rglob("*.json"))
    else:
        json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0}

    print(f"Found {len(json_files)} JSON file(s) to convert")
    print(f"Mode: {'Recursive' if recursive else 'Single directory'}")
    print(f"Delete originals: {'Yes' if delete_json else 'No'}")
    print(f"Validation: {'Enabled' if validate else 'Disabled'}")
    print("-" * 70)

    stats = {
        'total': len(json_files),
        'success': 0,
        'failed': 0,
        'total_json_size': 0,
        'total_npz_size': 0,
        'deleted': 0
    }

    # Process each file
    for json_path in tqdm(json_files, desc="Converting", disable=verbose):
        npz_path = json_path.with_suffix('.npz')

        # Track original size
        stats['total_json_size'] += json_path.stat().st_size

        # Convert
        success = convert_json_to_npz(json_path, npz_path, verbose=verbose)

        if success:
            stats['success'] += 1
            stats['total_npz_size'] += npz_path.stat().st_size

            # Validate if requested
            if validate:
                if verbose:
                    print(f"  Validating...", end=" ")

                is_valid = validate_conversion(json_path, npz_path)

                if not is_valid:
                    print(f"  FAILED validation")
                    stats['failed'] += 1
                    stats['success'] -= 1
                    npz_path.unlink()  # Delete invalid NPZ
                    continue
                elif verbose:
                    print("✓")

            # Delete JSON if requested
            if delete_json:
                json_path.unlink()
                stats['deleted'] += 1
                if verbose:
                    print(f"  Deleted {json_path.name}")
        else:
            stats['failed'] += 1

    # Print summary
    print("-" * 70)
    print("CONVERSION SUMMARY:")
    print(f"  Total files:    {stats['total']}")
    print(f"  Successful:     {stats['success']}")
    print(f"  Failed:         {stats['failed']}")

    if stats['success'] > 0:
        total_reduction = (1 - stats['total_npz_size'] / stats['total_json_size']) * 100
        space_saved = (stats['total_json_size'] - stats['total_npz_size']) / 1024 / 1024

        print(f"\nSTORAGE SAVINGS:")
        print(f"  Original size:  {stats['total_json_size'] / 1024 / 1024:.2f} MB")
        print(f"  New size:       {stats['total_npz_size'] / 1024 / 1024:.2f} MB")
        print(f"  Space saved:    {space_saved:.2f} MB ({total_reduction:.1f}% reduction)")

    if delete_json:
        print(f"\nDeleted {stats['deleted']} original JSON file(s)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON landmark files to compressed NPZ format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all JSON files in a directory
  python convert_json_to_npz.py --input-dir ./output

  # Convert recursively and delete originals
  python convert_json_to_npz.py --input-dir ./data --recursive --delete-json

  # Convert a single file
  python convert_json_to_npz.py --single-file ./output/video_landmarks.json

  # Convert without validation (faster)
  python convert_json_to_npz.py --input-dir ./output --no-validate
        """
    )

    parser.add_argument('--input-dir', type=str,
                        help='Directory containing JSON files to convert')

    parser.add_argument('--single-file', type=str,
                        help='Convert a single JSON file')

    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search subdirectories recursively')

    parser.add_argument('--delete-json', action='store_true',
                        help='Delete original JSON files after successful conversion')

    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation step (faster but less safe)')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output (show progress bar only)')

    args = parser.parse_args()

    # Validate arguments
    if not args.input_dir and not args.single_file:
        parser.error("Must specify either --input-dir or --single-file")

    if args.input_dir and args.single_file:
        parser.error("Cannot specify both --input-dir and --single-file")

    # Convert single file
    if args.single_file:
        json_path = Path(args.single_file)

        if not json_path.exists():
            print(f"ERROR: File not found: {json_path}")
            return 1

        if not json_path.suffix == '.json':
            print(f"ERROR: Not a JSON file: {json_path}")
            return 1

        npz_path = json_path.with_suffix('.npz')

        print(f"Converting {json_path.name}...")
        success = convert_json_to_npz(json_path, npz_path, verbose=True)

        if success and not args.no_validate:
            print("Validating...", end=" ")
            is_valid = validate_conversion(json_path, npz_path)
            if is_valid:
                print("✓")
            else:
                print("FAILED")
                npz_path.unlink()
                return 1

        if success and args.delete_json:
            json_path.unlink()
            print(f"Deleted {json_path.name}")

        return 0 if success else 1

    # Convert directory
    if args.input_dir:
        input_dir = Path(args.input_dir)

        if not input_dir.exists():
            print(f"ERROR: Directory not found: {input_dir}")
            return 1

        if not input_dir.is_dir():
            print(f"ERROR: Not a directory: {input_dir}")
            return 1

        stats = convert_directory(
            input_dir,
            recursive=args.recursive,
            delete_json=args.delete_json,
            validate=not args.no_validate,
            verbose=not args.quiet
        )

        return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())