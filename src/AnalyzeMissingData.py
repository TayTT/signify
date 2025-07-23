#!/usr/bin/env python3
"""
Missing Data Analysis Tool for Sign Language JSON Files

Analyzes missing landmark data patterns across JSON files to help understand
data quality and inform preprocessing strategies.

Usage:
    python src/AnalyzeMissingData.py --data_dir ./phoenix_dev --output missing_data_report.json

OR:
    from missing_data_analyzer import MissingDataAnalyzer

    analyzer = MissingDataAnalyzer("./phoenix_dev", "phoenix_analysis.json")

    results = analyzer.main()

    print(f"Mean missing data: {results['summary_statistics']['missing_data_percentages']['any_missing']['mean']:.1f}%")
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob


class MissingDataAnalyzer:
    """Analyzer for missing landmark data in sign language JSON files"""

    def __init__(self, data_directory: str, output_path: str = "missing_data_analysis.json"):
        """
        Initialize the missing data analyzer

        Args:
            data_directory: Directory containing JSON landmark files
            output_path: Where to save the analysis report
        """
        self.data_directory = Path(data_directory)
        self.output_path = Path(output_path)

        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

        # self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find all JSON files
        self.json_files = list(self.data_directory.glob("*.json"))

        if not self.json_files:
            raise ValueError(f"No JSON files found in {self.data_directory}")

        print(f"Initialized analyzer for {len(self.json_files)} JSON files")
        print(f"Output will be saved to: {self.output_path}")

    def analyze_all_files(self) -> Dict:
        """
        Analyze missing data patterns across all JSON files

        Returns:
            Dictionary containing the complete analysis
        """
        print(f"Analyzing missing data in {len(self.json_files)} JSON files...")

        # Initialize results structure
        analysis_results = {
            "metadata": {
                "analyzed_files": len(self.json_files),
                "analysis_date": str(pd.Timestamp.now()),
                "directory": str(self.data_directory),
                "successful_files": 0,
                "failed_files": 0
            },
            "per_file_analysis": {},
            "summary_statistics": {}
        }

        # Per-file analysis
        all_missing_percentages = {
            'left_hand': [],
            'right_hand': [],
            'face': [],
            'any_missing': []
        }

        # CHANGE THIS: Track filenames with sequence lengths
        sequence_length_data = []  # List of (filename, length) tuples
        total_missing_periods = 0
        successful_files = 0

        # Process each file
        for i, json_file in enumerate(self.json_files):
            try:
                print(f"Processing {i + 1}/{len(self.json_files)}: {json_file.name}")

                file_analysis = self._analyze_single_json_file(json_file)
                analysis_results["per_file_analysis"][json_file.name] = file_analysis

                # CHANGE THIS: Collect filename with length
                sequence_length_data.append((json_file.name, file_analysis['total_frames']))

                for component in ['left_hand', 'right_hand', 'face', 'any_missing']:
                    all_missing_percentages[component].append(
                        file_analysis['missing_data_summary'][component]['percentage']
                    )

                # Count missing periods
                for component in ['left_hand', 'right_hand', 'face']:
                    total_missing_periods += len(file_analysis['missing_periods'][component])

                successful_files += 1

            except Exception as e:
                print(f"ERROR analyzing {json_file.name}: {e}")
                analysis_results["per_file_analysis"][json_file.name] = {
                    "error": str(e),
                    "total_frames": 0
                }

        # Update metadata
        analysis_results["metadata"]["successful_files"] = successful_files
        analysis_results["metadata"]["failed_files"] = len(self.json_files) - successful_files

        # Calculate summary statistics
        analysis_results["summary_statistics"] = self._calculate_summary_statistics(
            sequence_length_data, all_missing_percentages, total_missing_periods  # CHANGE THIS: pass filename data
        )

        return analysis_results

    def _analyze_single_json_file(self, json_file_path: Path) -> Dict:
        """Analyze missing data patterns in a single JSON file"""

        with open(json_file_path, 'r') as f:
            data = json.load(f)

        frames_data = data.get('frames', {})
        total_frames = len(frames_data)

        if total_frames == 0:
            raise ValueError("No frames data found in JSON file")

        # Initialize counters
        missing_counts = {
            'left_hand': 0,
            'right_hand': 0,
            'face': 0,
            'any_missing': 0
        }

        # Track missing data per frame
        frame_missing_status = {
            'left_hand': [],
            'right_hand': [],
            'face': [],
            'any_missing': []
        }

        # Sort frames by frame number
        sorted_frames = sorted(frames_data.items(), key=lambda x: int(x[0]))

        for frame_key, frame_data in sorted_frames:
            # Check missing data - try different ways to detect it
            missing_data = frame_data.get('missing_data', {})

            # Method 1: Use explicit missing_data field if available
            if missing_data:
                left_hand_missing = missing_data.get('left_hand_missing', False)
                right_hand_missing = missing_data.get('right_hand_missing', False)
                face_missing = missing_data.get('face_missing', False)
                any_missing = missing_data.get('any_missing', False)
            else:
                # Method 2: Infer from landmark data structure
                hands_data = frame_data.get('hands', {})
                face_data = frame_data.get('face', {})

                left_hand_missing = self._is_hand_data_missing(hands_data.get('left_hand', {}))
                right_hand_missing = self._is_hand_data_missing(hands_data.get('right_hand', {}))
                face_missing = self._is_face_data_missing(face_data)
                any_missing = left_hand_missing or right_hand_missing or face_missing

            # Count missing data
            if left_hand_missing:
                missing_counts['left_hand'] += 1
            if right_hand_missing:
                missing_counts['right_hand'] += 1
            if face_missing:
                missing_counts['face'] += 1
            if any_missing:
                missing_counts['any_missing'] += 1

            # Track per-frame status
            frame_missing_status['left_hand'].append(left_hand_missing)
            frame_missing_status['right_hand'].append(right_hand_missing)
            frame_missing_status['face'].append(face_missing)
            frame_missing_status['any_missing'].append(any_missing)

        # Find missing periods (contiguous segments)
        missing_periods = {}
        for component in ['left_hand', 'right_hand', 'face']:
            missing_periods[component] = self._find_missing_periods(
                frame_missing_status[component]
            )

        # Calculate percentages
        missing_percentages = {
            component: (count / total_frames * 100) if total_frames > 0 else 0
            for component, count in missing_counts.items()
        }

        return {
            "filename": json_file_path.name,
            "total_frames": total_frames,
            "missing_data_summary": {
                component: {
                    "missing_frames": missing_counts[component],
                    "percentage": missing_percentages[component]
                }
                for component in missing_counts.keys()
            },
            "missing_periods": missing_periods,
            "missing_period_stats": {
                component: {
                    "num_periods": len(periods),
                    "avg_period_length": np.mean([p['length'] for p in periods]) if periods else 0,
                    "max_period_length": max([p['length'] for p in periods]) if periods else 0,
                    "min_period_length": min([p['length'] for p in periods]) if periods else 0
                }
                for component, periods in missing_periods.items()
            }
        }

    def _is_hand_data_missing(self, hand_data) -> bool:
        """Check if hand data is missing or empty"""
        if not hand_data:
            return True

        if isinstance(hand_data, dict):
            landmarks = hand_data.get('landmarks', [])
            confidence = hand_data.get('confidence', 0)
            return not landmarks or confidence < 0.1
        elif isinstance(hand_data, list):
            return len(hand_data) == 0

        return True

    def _is_face_data_missing(self, face_data) -> bool:
        """Check if face data is missing or empty"""
        if not face_data:
            return True

        if isinstance(face_data, dict):
            landmarks = face_data.get('all_landmarks', [])
            return not landmarks

        return True

    def _find_missing_periods(self, missing_status_list: List[bool]) -> List[Dict]:
        """Find contiguous periods of missing data"""
        periods = []
        in_missing_period = False
        period_start = 0

        for i, is_missing in enumerate(missing_status_list):
            if is_missing and not in_missing_period:
                # Start of missing period
                period_start = i
                in_missing_period = True
            elif not is_missing and in_missing_period:
                # End of missing period
                period_length = i - period_start
                periods.append({
                    "start_frame": period_start,
                    "end_frame": i - 1,
                    "length": period_length
                })
                in_missing_period = False

        # Handle period that goes to the end
        if in_missing_period:
            period_length = len(missing_status_list) - period_start
            periods.append({
                "start_frame": period_start,
                "end_frame": len(missing_status_list) - 1,
                "length": period_length
            })

        return periods

    def _calculate_summary_statistics(self, sequence_length_data: List[Tuple[str, int]],
                                      all_missing_percentages: Dict,
                                      total_missing_periods: int) -> Dict:
        """Calculate overall summary statistics"""

        # Extract just the lengths for calculations
        all_sequence_lengths = [length for _, length in sequence_length_data]

        summary = {
            "sequence_lengths": {
                "mean": np.mean(all_sequence_lengths) if all_sequence_lengths else 0,
                "min": np.min(all_sequence_lengths) if all_sequence_lengths else 0,
                "max": np.max(all_sequence_lengths) if all_sequence_lengths else 0,
                "median": np.median(all_sequence_lengths) if all_sequence_lengths else 0,
                "std": np.std(all_sequence_lengths) if all_sequence_lengths else 0
            },
            "missing_data_percentages": {},
            "total_missing_periods": total_missing_periods,
            "files_with_any_missing": sum(1 for p in all_missing_percentages.get('any_missing', []) if p > 0)
        }

        # ADD THIS: Find files with min/max frames
        if sequence_length_data:
            # Find min and max
            min_length = min(all_sequence_lengths)
            max_length = max(all_sequence_lengths)

            # Find filenames corresponding to min/max
            min_files = [filename for filename, length in sequence_length_data if length == min_length]
            max_files = [filename for filename, length in sequence_length_data if length == max_length]

            # Add to metadata (take first file if there are ties)
            summary["sequence_lengths"]["min_frames_file"] = min_files[0]
            summary["sequence_lengths"]["max_frames_file"] = max_files[0]

        # Calculate missing data statistics
        for component, percentages in all_missing_percentages.items():
            if percentages:
                summary["missing_data_percentages"][component] = {
                    "mean": np.mean(percentages),
                    "min": np.min(percentages),
                    "max": np.max(percentages),
                    "median": np.median(percentages),
                    "std": np.std(percentages)
                }
            else:
                summary["missing_data_percentages"][component] = {
                    "mean": 0, "min": 0, "max": 0, "median": 0, "std": 0
                }

        return summary

    def save_analysis(self, analysis_results: Dict):
        """Save analysis results to JSON file"""
        with open(self.output_path, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)

        print(f"Analysis saved to: {self.output_path}")

    def print_analysis_summary(self, analysis_results: Dict):
        """Print a human-readable summary of the analysis"""
        metadata = analysis_results["metadata"]
        summary = analysis_results["summary_statistics"]

        print(f"\n{'=' * 50}")
        print(f"MISSING DATA ANALYSIS SUMMARY")
        print(f"{'=' * 50}")

        print(f"\nDataset Information:")
        print(f"   Directory: {metadata['directory']}")
        print(f"   Files analyzed: {metadata['successful_files']}/{metadata['analyzed_files']}")
        if metadata['failed_files'] > 0:
            print(f"   Failed files: {metadata['failed_files']}")

        print(f"\nSequence Length Statistics:")
        seq_stats = summary["sequence_lengths"]
        print(f"   Mean length: {seq_stats['mean']:.1f} frames")
        print(f"   Range: {seq_stats['min']:.0f} - {seq_stats['max']:.0f} frames")
        print(f"   Median: {seq_stats['median']:.1f} frames")
        print(f"   Std dev: {seq_stats['std']:.1f} frames")

        # ADD THIS: Show files with min/max frames
        if 'min_frames_file' in seq_stats and 'max_frames_file' in seq_stats:
            print(f"   Shortest sequence: {seq_stats['min_frames_file']} ({seq_stats['min']:.0f} frames)")
            print(f"   Longest sequence: {seq_stats['max_frames_file']} ({seq_stats['max']:.0f} frames)")

        print(f"\nMissing Data Statistics:")
        missing_stats = summary["missing_data_percentages"]

        for component in ['left_hand', 'right_hand', 'face', 'any_missing']:
            if component in missing_stats:
                stats = missing_stats[component]
                component_name = component.replace('_', ' ').title()
                print(f"   {component_name}:")
                print(f"     Mean missing: {stats['mean']:.1f}%")
                print(f"     Range: {stats['min']:.1f}% - {stats['max']:.1f}%")
                print(f"     Median: {stats['median']:.1f}%")
                print(f"     Std dev: {stats['std']:.1f}%")

        print(f"\nOverall Statistics:")
        print(f"   Total missing periods: {summary['total_missing_periods']}")
        print(f"   Files with any missing data: {summary['files_with_any_missing']}/{metadata['successful_files']}")

        # Show problematic files
        self._print_problematic_files(analysis_results)

    def _print_problematic_files(self, analysis_results: Dict):
        """Print files with highest missing data percentages"""
        print(f"\nFiles with Highest Missing Data:")

        per_file = analysis_results["per_file_analysis"]

        # Sort files by any_missing percentage
        valid_files = [
            (filename, data) for filename, data in per_file.items()
            if 'missing_data_summary' in data and 'error' not in data
        ]

        if not valid_files:
            print("   No valid files to analyze")
            return

        sorted_files = sorted(
            valid_files,
            key=lambda x: x[1]['missing_data_summary']['any_missing']['percentage'],
            reverse=True
        )

        # Show top 10 worst files
        for i, (filename, data) in enumerate(sorted_files[:10]):
            any_missing_pct = data['missing_data_summary']['any_missing']['percentage']
            total_frames = data['total_frames']
            left_pct = data['missing_data_summary']['left_hand']['percentage']
            right_pct = data['missing_data_summary']['right_hand']['percentage']
            face_pct = data['missing_data_summary']['face']['percentage']

            print(f"   {i + 1:2d}. {filename}")
            print(f"       Overall: {any_missing_pct:.1f}% missing ({total_frames} frames)")
            print(f"       L:{left_pct:.0f}% R:{right_pct:.0f}% F:{face_pct:.0f}%")

        if len(sorted_files) > 10:
            print(f"   ... and {len(sorted_files) - 10} more files")

    def main(self):
        """Main analysis pipeline"""
        print(f"Starting missing data analysis...")
        print(f"Data directory: {self.data_directory}")
        print(f"Output file: {self.output_path}")

        try:
            # Run the analysis
            analysis_results = self.analyze_all_files()

            # Save results
            self.save_analysis(analysis_results)

            # Print summary
            self.print_analysis_summary(analysis_results)

            print(f"\nAnalysis completed successfully!")
            print(f"Full results saved to: {self.output_path}")

            return analysis_results

        except Exception as e:
            print(f"Analysis failed: {e}")
            raise


def main():
    """Command line interface for missing data analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze missing landmark data in sign language JSON files',
        epilog='''
Examples:
  python missing_data_analyzer.py --data_dir ./phoenix_dev
  python missing_data_analyzer.py --data_dir ./my_data --output my_analysis.json
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing JSON landmark files')
    parser.add_argument('--output', type=str, default='missing_data_analysis.json',
                        help='Output JSON file for analysis results')

    args = parser.parse_args()

    try:
        # Initialize and run analyzer
        analyzer = MissingDataAnalyzer(args.data_dir, args.output)
        results = analyzer.main()

        return results

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()