#!/usr/bin/env python3
"""
Video Trimming Utility for Sign Language JSON Data

This script trims JSON landmark files to remove inactive frames at the beginning
and end of videos, keeping only frames from first hand detection to last hand detection.

Usage:
    python trim_videos.py --input_dir ./original_jsons --output_dir ./trimmed_jsons
    python trim_videos.py --input_dir ./phoenix_data --output_dir ./phoenix_trimmed --recursive
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import shutil


class VideoTrimmer:
    """Standalone video trimmer for JSON landmark data"""

    def __init__(self, min_sequence_length: int = 5):
        """
        Initialize trimmer

        Args:
            min_sequence_length: Minimum number of frames to keep after trimming
        """
        self.min_sequence_length = min_sequence_length
        self.stats = {
            'total_files': 0,
            'successfully_trimmed': 0,
            'no_detections': 0,
            'too_short_after_trim': 0,
            'no_trimming_needed': 0,
            'total_frames_removed': 0,
            'total_original_frames': 0
        }

    def _has_hand_detection(self, hand_data: Dict) -> bool:
        """Check if hand data contains valid landmarks"""
        if not hand_data:
            return False

        landmarks = hand_data.get('landmarks', [])
        if not landmarks:
            return False

        # Check if landmarks have non-zero coordinates
        for landmark in landmarks:
            if isinstance(landmark, dict):
                if landmark.get('x', 0) != 0 or landmark.get('y', 0) != 0:
                    return True
            elif hasattr(landmark, 'x'):
                if landmark.x != 0 or landmark.y != 0:
                    return True

        return False

    def _find_detection_boundaries(self, frames_data: Dict) -> Tuple[Optional[int], Optional[int]]:
        """
        Find first and last frame indices with hand detections

        Args:
            frames_data: Dictionary of frame data from JSON

        Returns:
            Tuple of (first_detection_index, last_detection_index) or (None, None) if no detections
        """
        frame_keys = sorted(frames_data.keys(), key=int)
        first_detection = None
        last_detection = None

        for i, frame_key in enumerate(frame_keys):
            frame_data = frames_data[frame_key]
            hands = frame_data.get('hands', {})

            # Check if either hand is detected
            has_left = self._has_hand_detection(hands.get('left_hand', {}))
            has_right = self._has_hand_detection(hands.get('right_hand', {}))

            if has_left or has_right:
                if first_detection is None:
                    first_detection = i
                last_detection = i

        return first_detection, last_detection

    def trim_json_file(self, json_path: Path, output_path: Path, verbose: bool = True) -> Dict[str, Any]:
        """
        Trim a single JSON file and save the result

        Args:
            json_path: Path to input JSON file
            output_path: Path to save trimmed JSON file
            verbose: Print detailed information

        Returns:
            Dictionary with trimming statistics for this file
        """
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)

            if 'frames' not in data:
                if verbose:
                    print(f"  ERROR: Invalid JSON format (no 'frames' key): {json_path.name}")
                return {'error': 'invalid_format'}

            frames_data = data['frames']
            original_frame_count = len(frames_data)
            self.stats['total_original_frames'] += original_frame_count

            if verbose:
                print(f"  Original frames: {original_frame_count}")

            # Find detection boundaries
            first_detection, last_detection = self._find_detection_boundaries(frames_data)

            if first_detection is None or last_detection is None:
                if verbose:
                    print(f"  WARNING: No hand detections found - copying original file")
                self.stats['no_detections'] += 1

                # Copy original file unchanged
                shutil.copy2(json_path, output_path)
                return {
                    'original_frames': original_frame_count,
                    'trimmed_frames': original_frame_count,
                    'frames_removed': 0,
                    'status': 'no_detections'
                }

            # Check if trimming would make sequence too short
            detection_span = last_detection - first_detection + 1
            if detection_span < self.min_sequence_length:
                if verbose:
                    print(f"  WARNING: Detection span too short ({detection_span} frames) - copying original")
                self.stats['too_short_after_trim'] += 1

                # Copy original file unchanged
                shutil.copy2(json_path, output_path)
                return {
                    'original_frames': original_frame_count,
                    'trimmed_frames': original_frame_count,
                    'frames_removed': 0,
                    'status': 'too_short_after_trim'
                }

            # Check if any trimming is actually needed
            frames_to_remove = first_detection + (original_frame_count - last_detection - 1)
            if frames_to_remove == 0:
                if verbose:
                    print(f"  No trimming needed - detections span entire video")
                self.stats['no_trimming_needed'] += 1

                # Copy original file unchanged
                shutil.copy2(json_path, output_path)
                return {
                    'original_frames': original_frame_count,
                    'trimmed_frames': original_frame_count,
                    'frames_removed': 0,
                    'status': 'no_trimming_needed'
                }

            # Apply trimming
            frame_keys = sorted(frames_data.keys(), key=int)
            frames_to_keep = frame_keys[first_detection:last_detection + 1]

            # Create trimmed frame data with renumbered keys
            trimmed_frames_data = {}
            for new_idx, original_key in enumerate(frames_to_keep):
                new_key = str(new_idx)  # Renumber from 0
                trimmed_frames_data[new_key] = frames_data[original_key]

            # Update metadata
            trimmed_data = data.copy()
            trimmed_data['frames'] = trimmed_frames_data

            # Add trimming information to metadata
            if 'metadata' not in trimmed_data:
                trimmed_data['metadata'] = {}

            trimmed_data['metadata']['trimming_applied'] = True
            trimmed_data['metadata']['original_frame_count'] = original_frame_count
            trimmed_data['metadata']['trimmed_frame_count'] = len(trimmed_frames_data)
            trimmed_data['metadata']['frames_removed_start'] = first_detection
            trimmed_data['metadata']['frames_removed_end'] = original_frame_count - last_detection - 1
            trimmed_data['metadata']['first_detection_frame'] = first_detection
            trimmed_data['metadata']['last_detection_frame'] = last_detection

            # Save trimmed JSON
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(trimmed_data, f, indent=2)

            trimmed_frame_count = len(trimmed_frames_data)
            frames_removed = original_frame_count - trimmed_frame_count

            if verbose:
                print(f"  Trimmed: {original_frame_count} → {trimmed_frame_count} frames "
                      f"(removed {first_detection} from start, {original_frame_count - last_detection - 1} from end)")

            self.stats['successfully_trimmed'] += 1
            self.stats['total_frames_removed'] += frames_removed

            return {
                'original_frames': original_frame_count,
                'trimmed_frames': trimmed_frame_count,
                'frames_removed': frames_removed,
                'frames_removed_start': first_detection,
                'frames_removed_end': original_frame_count - last_detection - 1,
                'status': 'successfully_trimmed'
            }

        except Exception as e:
            if verbose:
                print(f"  ERROR: Failed to process {json_path.name}: {str(e)}")
            return {'error': str(e)}

    def trim_directory(self, input_dir: Path, output_dir: Path, recursive: bool = False,
                       file_pattern: str = "*.json") -> None:
        """
        Trim all JSON files in a directory

        Args:
            input_dir: Directory containing original JSON files
            output_dir: Directory to save trimmed JSON files
            recursive: Whether to process subdirectories
            file_pattern: Pattern to match JSON files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            return

        # Find JSON files
        if recursive:
            json_files = list(input_dir.rglob(file_pattern))
        else:
            json_files = list(input_dir.glob(file_pattern))

        if not json_files:
            print(f"ERROR: No JSON files found in {input_dir}")
            return

        print(f"Found {len(json_files)} JSON files to process")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Recursive: {recursive}")
        print("-" * 60)

        # Process each file
        for i, json_file in enumerate(json_files):
            # Calculate relative path to preserve directory structure
            relative_path = json_file.relative_to(input_dir)
            output_file = output_dir / relative_path

            print(f"[{i + 1}/{len(json_files)}] Processing: {relative_path}")

            result = self.trim_json_file(json_file, output_file, verbose=True)

            if 'error' in result:
                print(f"  FAILED: {result['error']}")

        # Print summary statistics
        self._print_summary_statistics()

    def _print_summary_statistics(self):
        """Print summary of trimming operation"""
        print("\n" + "=" * 60)
        print("TRIMMING SUMMARY")
        print("=" * 60)

        total_files = self.stats['total_files'] = (
                self.stats['successfully_trimmed'] +
                self.stats['no_detections'] +
                self.stats['too_short_after_trim'] +
                self.stats['no_trimming_needed']
        )

        print(f"Total files processed: {total_files}")
        print(f"Successfully trimmed: {self.stats['successfully_trimmed']}")
        print(f"No hand detections: {self.stats['no_detections']}")
        print(f"Too short after trim: {self.stats['too_short_after_trim']}")
        print(f"No trimming needed: {self.stats['no_trimming_needed']}")

        if self.stats['total_original_frames'] > 0:
            removal_percentage = (self.stats['total_frames_removed'] / self.stats['total_original_frames']) * 100
            print(f"\nFrame Statistics:")
            print(f"Total original frames: {self.stats['total_original_frames']:,}")
            print(f"Total frames removed: {self.stats['total_frames_removed']:,}")
            print(f"Frames removed: {removal_percentage:.1f}%")

            if self.stats['successfully_trimmed'] > 0:
                avg_removed_per_file = self.stats['total_frames_removed'] / self.stats['successfully_trimmed']
                print(f"Average frames removed per trimmed file: {avg_removed_per_file:.1f}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Trim JSON landmark files to remove inactive segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trim all JSON files in a directory
  python trim_videos.py --input_dir ./output --output_dir ./output_trimmed

  # Recursively process subdirectories
  python trim_videos.py --input_dir ./phoenix_data --output_dir ./phoenix_trimmed --recursive

  # Custom minimum sequence length
  python trim_videos.py --input_dir ./data --output_dir ./data_trimmed --min_length 10

  # Process specific file pattern
  python trim_videos.py --input_dir ./data --output_dir ./data_trimmed --pattern "*_landmarks.json"
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing original JSON files')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trimmed JSON files')

    parser.add_argument('--recursive', action='store_true',
                        help='Process subdirectories recursively')

    parser.add_argument('--pattern', type=str, default='*.json',
                        help='File pattern to match (default: *.json)')

    parser.add_argument('--min_length', type=int, default=5,
                        help='Minimum sequence length after trimming (default: 5)')

    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be trimmed without actually saving files')

    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files in output directory')

    args = parser.parse_args()

    # Validate directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    if output_dir.exists() and not args.overwrite and not args.dry_run:
        response = input(f"Output directory {output_dir} already exists. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return

    # Initialize trimmer
    trimmer = VideoTrimmer(min_sequence_length=args.min_length)

    if args.dry_run:
        print("DRY RUN MODE - No files will be saved")
        print("-" * 40)

    # Process files
    if args.dry_run:
        trimmer.analyze_directory(input_dir, recursive=args.recursive, file_pattern=args.pattern)
    else:
        trimmer.trim_directory(input_dir, output_dir, recursive=args.recursive, file_pattern=args.pattern)


class VideoTrimmer:
    """Standalone video trimmer for JSON landmark data"""

    def __init__(self, min_sequence_length: int = 5):
        self.min_sequence_length = min_sequence_length
        self.stats = {
            'total_files': 0,
            'successfully_trimmed': 0,
            'no_detections': 0,
            'too_short_after_trim': 0,
            'no_trimming_needed': 0,
            'total_frames_removed': 0,
            'total_original_frames': 0
        }

    def _has_hand_detection(self, hand_data: Dict) -> bool:
        """Check if hand data contains valid landmarks"""
        if not hand_data:
            return False

        landmarks = hand_data.get('landmarks', [])
        if not landmarks:
            return False

        # Check if landmarks have non-zero coordinates
        for landmark in landmarks:
            if isinstance(landmark, dict):
                if landmark.get('x', 0) != 0 or landmark.get('y', 0) != 0:
                    return True
            elif hasattr(landmark, 'x'):
                if landmark.x != 0 or landmark.y != 0:
                    return True

        return False

    def _find_detection_boundaries(self, frames_data: Dict) -> Tuple[Optional[int], Optional[int]]:
        """Find first and last frame indices with hand detections"""
        frame_keys = sorted(frames_data.keys(), key=int)
        first_detection = None
        last_detection = None

        for i, frame_key in enumerate(frame_keys):
            frame_data = frames_data[frame_key]
            hands = frame_data.get('hands', {})

            # Check if either hand is detected
            has_left = self._has_hand_detection(hands.get('left_hand', {}))
            has_right = self._has_hand_detection(hands.get('right_hand', {}))

            if has_left or has_right:
                if first_detection is None:
                    first_detection = i
                last_detection = i

        return first_detection, last_detection

    def trim_json_file(self, json_path: Path, output_path: Path, verbose: bool = True) -> Dict[str, Any]:
        """Trim a single JSON file and save the result"""
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)

            if 'frames' not in data:
                if verbose:
                    print(f"  ERROR: Invalid JSON format (no 'frames' key): {json_path.name}")
                return {'error': 'invalid_format'}

            frames_data = data['frames']
            original_frame_count = len(frames_data)
            self.stats['total_original_frames'] += original_frame_count

            if verbose:
                print(f"  Original frames: {original_frame_count}")

            # Find detection boundaries
            first_detection, last_detection = self._find_detection_boundaries(frames_data)

            if first_detection is None or last_detection is None:
                if verbose:
                    print(f"  WARNING: No hand detections found - copying original file")
                self.stats['no_detections'] += 1

                # Copy original file unchanged
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(json_path, output_path)
                return {
                    'original_frames': original_frame_count,
                    'trimmed_frames': original_frame_count,
                    'frames_removed': 0,
                    'status': 'no_detections'
                }

            # Check if trimming would make sequence too short
            detection_span = last_detection - first_detection + 1
            if detection_span < self.min_sequence_length:
                if verbose:
                    print(f"  WARNING: Detection span too short ({detection_span} frames) - copying original")
                self.stats['too_short_after_trim'] += 1

                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(json_path, output_path)
                return {
                    'original_frames': original_frame_count,
                    'trimmed_frames': original_frame_count,
                    'frames_removed': 0,
                    'status': 'too_short_after_trim'
                }

            # Check if any trimming is actually needed
            frames_to_remove = first_detection + (original_frame_count - last_detection - 1)
            if frames_to_remove == 0:
                if verbose:
                    print(f"  No trimming needed - detections span entire video")
                self.stats['no_trimming_needed'] += 1

                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(json_path, output_path)
                return {
                    'original_frames': original_frame_count,
                    'trimmed_frames': original_frame_count,
                    'frames_removed': 0,
                    'status': 'no_trimming_needed'
                }

            # Apply trimming
            frame_keys = sorted(frames_data.keys(), key=int)
            frames_to_keep = frame_keys[first_detection:last_detection + 1]

            # Create trimmed frame data with renumbered keys
            trimmed_frames_data = {}
            for new_idx, original_key in enumerate(frames_to_keep):
                new_key = str(new_idx)  # Renumber from 0
                trimmed_frames_data[new_key] = frames_data[original_key]

            # Update metadata
            trimmed_data = data.copy()
            trimmed_data['frames'] = trimmed_frames_data

            # Add trimming information to metadata
            if 'metadata' not in trimmed_data:
                trimmed_data['metadata'] = {}

            trimmed_data['metadata']['trimming_applied'] = True
            trimmed_data['metadata']['original_frame_count'] = original_frame_count
            trimmed_data['metadata']['trimmed_frame_count'] = len(trimmed_frames_data)
            trimmed_data['metadata']['frames_removed_start'] = first_detection
            trimmed_data['metadata']['frames_removed_end'] = original_frame_count - last_detection - 1
            trimmed_data['metadata']['first_detection_frame'] = first_detection
            trimmed_data['metadata']['last_detection_frame'] = last_detection

            # Save trimmed JSON
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(trimmed_data, f, indent=2)

            trimmed_frame_count = len(trimmed_frames_data)
            frames_removed = original_frame_count - trimmed_frame_count

            if verbose:
                print(f"  SUCCESS: {original_frame_count} → {trimmed_frame_count} frames "
                      f"(removed {first_detection} from start, {original_frame_count - last_detection - 1} from end)")

            self.stats['successfully_trimmed'] += 1
            self.stats['total_frames_removed'] += frames_removed

            return {
                'original_frames': original_frame_count,
                'trimmed_frames': trimmed_frame_count,
                'frames_removed': frames_removed,
                'frames_removed_start': first_detection,
                'frames_removed_end': original_frame_count - last_detection - 1,
                'status': 'successfully_trimmed'
            }

        except Exception as e:
            if verbose:
                print(f"  ERROR: Failed to process {json_path.name}: {str(e)}")
            return {'error': str(e)}

    def trim_directory(self, input_dir: Path, output_dir: Path, recursive: bool = False,
                       file_pattern: str = "*.json") -> None:
        """Trim all JSON files in a directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            return

        # Find JSON files
        if recursive:
            json_files = list(input_dir.rglob(file_pattern))
        else:
            json_files = list(input_dir.glob(file_pattern))

        if not json_files:
            print(f"ERROR: No JSON files found in {input_dir}")
            print(f"Pattern used: {file_pattern}")
            print(f"Recursive: {recursive}")
            return

        print(f"Found {len(json_files)} JSON files to process")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Recursive: {recursive}")
        print(f"File pattern: {file_pattern}")
        print("-" * 60)

        # Process each file
        for i, json_file in enumerate(json_files):
            # Calculate relative path to preserve directory structure
            relative_path = json_file.relative_to(input_dir)
            output_file = output_dir / relative_path

            print(f"[{i + 1}/{len(json_files)}] Processing: {relative_path}")

            result = self.trim_json_file(json_file, output_file, verbose=True)

            if 'error' in result:
                print(f"  FAILED: {result['error']}")

        # Print summary statistics
        self._print_summary_statistics()

    def analyze_directory(self, input_dir: Path, recursive: bool = False,
                          file_pattern: str = "*.json") -> None:
        """Analyze what would be trimmed without actually saving files (dry run)"""
        input_dir = Path(input_dir)

        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            return

        # Find JSON files
        if recursive:
            json_files = list(input_dir.rglob(file_pattern))
        else:
            json_files = list(input_dir.glob(file_pattern))

        if not json_files:
            print(f"ERROR: No JSON files found in {input_dir}")
            return

        print(f"Found {len(json_files)} JSON files to analyze")
        print(f"Input directory: {input_dir}")
        print("-" * 60)

        # Analyze each file
        for i, json_file in enumerate(json_files):
            relative_path = json_file.relative_to(input_dir)
            print(f"[{i + 1}/{len(json_files)}] Analyzing: {relative_path}")

            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if 'frames' not in data:
                    print(f"  ERROR: Invalid JSON format")
                    continue

                frames_data = data['frames']
                original_count = len(frames_data)

                first_detection, last_detection = self._find_detection_boundaries(frames_data)

                if first_detection is None:
                    print(f"  No hand detections found ({original_count} frames)")
                    self.stats['no_detections'] += 1
                else:
                    detection_span = last_detection - first_detection + 1
                    frames_to_remove_start = first_detection
                    frames_to_remove_end = original_count - last_detection - 1
                    total_to_remove = frames_to_remove_start + frames_to_remove_end

                    if detection_span < self.min_sequence_length:
                        print(f"  Detection span too short: {detection_span} frames")
                        self.stats['too_short_after_trim'] += 1
                    elif total_to_remove == 0:
                        print(f"  No trimming needed - full video has detections")
                        self.stats['no_trimming_needed'] += 1
                    else:
                        print(f"  Would trim: {original_count} → {detection_span} frames "
                              f"(remove {frames_to_remove_start} from start, {frames_to_remove_end} from end)")
                        self.stats['successfully_trimmed'] += 1
                        self.stats['total_frames_removed'] += total_to_remove

                self.stats['total_original_frames'] += original_count

            except Exception as e:
                print(f"  ERROR: {str(e)}")

        # Print summary
        self._print_summary_statistics()

    def _print_summary_statistics(self):
        """Print summary of trimming operation"""
        print("\n" + "=" * 60)
        print("TRIMMING SUMMARY")
        print("=" * 60)

        total_files = (
                self.stats['successfully_trimmed'] +
                self.stats['no_detections'] +
                self.stats['too_short_after_trim'] +
                self.stats['no_trimming_needed']
        )

        print(f"Total files analyzed: {total_files}")
        print(f"Would be trimmed: {self.stats['successfully_trimmed']}")
        print(f"No hand detections: {self.stats['no_detections']}")
        print(f"Too short after trim: {self.stats['too_short_after_trim']}")
        print(f"No trimming needed: {self.stats['no_trimming_needed']}")

        if self.stats['total_original_frames'] > 0:
            removal_percentage = (self.stats['total_frames_removed'] / self.stats['total_original_frames']) * 100
            print(f"\nFrame Statistics:")
            print(f"Total original frames: {self.stats['total_original_frames']:,}")
            print(f"Total frames to remove: {self.stats['total_frames_removed']:,}")
            print(f"Reduction: {removal_percentage:.1f}%")

            if self.stats['successfully_trimmed'] > 0:
                avg_removed_per_file = self.stats['total_frames_removed'] / self.stats['successfully_trimmed']
                print(f"Average frames removed per file: {avg_removed_per_file:.1f}")


if __name__ == "__main__":
    main()