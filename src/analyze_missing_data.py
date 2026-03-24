#!/usr/bin/env python3
"""
Missing Data Analysis Tool for Sign Language JSON Files

Analyzes missing landmark data patterns across raw JSON landmark files to
assess data quality and optionally produce a filtered annotations CSV.

<<<<<<< HEAD:src/analyze_missing_data.py
Usage:
    python src/analyze_missing_data.py --data_dir ./phoenix_dev --output missing_data_report.json

=======
Usage (directory scan only):
    python analyze_missing_data.py --data_dir ./phoenix_jsons/train --output report.json

Usage (annotations-aware, with filtered export):
    python analyze_missing_data.py
        --data_dir ./phoenix_jsons/train
        --annotations_path ./data/annotations_phoenix/train_corpus.csv
        --output quality_report.json
        --export_filtered filtered_train.csv
        --max_hand_pct 30.0
        --max_face_pct 50.0
>>>>>>> origin/data-quality:src/AnalyzeMissingData.py
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MissingDataAnalyzer:
    """Analyzer for missing landmark data in raw sign language JSON files"""

    def __init__(self, data_directory: str, output_path: str = "missing_data_analysis.json"):
        self.data_directory = Path(data_directory)
        self.output_path = Path(output_path)

        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

        # raw JSON only - npz are preprocessed tensors, no frame-level landmark info
        self.json_files = list(self.data_directory.glob("*.json"))

        if not self.json_files:
            raise ValueError(f"No JSON files found in {self.data_directory}")

        print(f"Initialized analyzer for {len(self.json_files)} JSON files")
        print(f"Output will be saved to: {self.output_path}")

    def analyze_all_files(self) -> Dict:
        """Analyze missing data patterns across all JSON files"""
        print(f"Analyzing {len(self.json_files)} files...")

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

        all_missing_percentages = {
            "left_hand": [],
            "right_hand": [],
            "face": [],
            "any_missing": [],
            "interior_hand_missing": []
        }
<<<<<<< HEAD:src/analyze_missing_data.py

        # Track filenames with sequence lengths
        sequence_length_data = []  # List of (filename, length) tuples
=======
        sequence_length_data = []
>>>>>>> origin/data-quality:src/AnalyzeMissingData.py
        total_missing_periods = 0
        successful_files = 0

        for i, json_file in enumerate(self.json_files):
            try:
                print(f"Processing {i + 1}/{len(self.json_files)}: {json_file.name}")
                file_analysis = self._analyze_single_json_file(json_file)
                analysis_results["per_file_analysis"][json_file.name] = file_analysis
                sequence_length_data.append((json_file.name, file_analysis["total_frames"]))

<<<<<<< HEAD:src/analyze_missing_data.py
                # Collect filename with length
                sequence_length_data.append((json_file.name, file_analysis['total_frames']))

                for component in ['left_hand', 'right_hand', 'face', 'any_missing']:
=======
                for component in ["left_hand", "right_hand", "face", "any_missing"]:
>>>>>>> origin/data-quality:src/AnalyzeMissingData.py
                    all_missing_percentages[component].append(
                        file_analysis["missing_data_summary"][component]["percentage"]
                    )
                all_missing_percentages["interior_hand_missing"].append(
                    file_analysis["interior_hand_missing_pct"]
                )

                for component in ["left_hand", "right_hand", "face"]:
                    total_missing_periods += len(file_analysis["missing_periods"][component])

                successful_files += 1

            except Exception as e:
                print(f"ERROR analyzing {json_file.name}: {e}")
                analysis_results["per_file_analysis"][json_file.name] = {
                    "error": str(e),
                    "total_frames": 0
                }

        analysis_results["metadata"]["successful_files"] = successful_files
        analysis_results["metadata"]["failed_files"] = len(self.json_files) - successful_files
        analysis_results["summary_statistics"] = self._calculate_summary_statistics(
<<<<<<< HEAD:src/analyze_missing_data.py
            sequence_length_data, all_missing_percentages, total_missing_periods  # pass filename data
=======
            sequence_length_data, all_missing_percentages, total_missing_periods
>>>>>>> origin/data-quality:src/AnalyzeMissingData.py
        )
        return analysis_results

    def _analyze_single_json_file(self, json_file_path: Path) -> Dict:
        """Analyze missing data patterns in a single raw JSON landmark file"""
        with open(json_file_path, "r") as f:
            data = json.load(f)

        frames_data = data.get("frames", {})
        total_frames = len(frames_data)

        if total_frames == 0:
            raise ValueError("No frames data found in JSON file")

        missing_counts = {
            "left_hand": 0,
            "right_hand": 0,
            "face": 0,
            "any_missing": 0
        }
        frame_missing_status = {
            "left_hand": [],
            "right_hand": [],
            "face": [],
            "any_missing": []
        }

        sorted_frames = sorted(frames_data.items(), key=lambda x: int(x[0]))

        for frame_key, frame_data in sorted_frames:
            missing_data = frame_data.get("missing_data", {})

            if missing_data:
                left_hand_missing = missing_data.get("left_hand_missing", False)
                right_hand_missing = missing_data.get("right_hand_missing", False)
                face_missing = missing_data.get("face_missing", False)
                any_missing = missing_data.get("any_missing", False)
            else:
                hands_data = frame_data.get("hands", {})
                face_data = frame_data.get("face", {})
                left_hand_missing = self._is_hand_data_missing(hands_data.get("left_hand", {}))
                right_hand_missing = self._is_hand_data_missing(hands_data.get("right_hand", {}))
                face_missing = self._is_face_data_missing(face_data)
                any_missing = left_hand_missing or right_hand_missing or face_missing

            if left_hand_missing:
                missing_counts["left_hand"] += 1
            if right_hand_missing:
                missing_counts["right_hand"] += 1
            if face_missing:
                missing_counts["face"] += 1
            if any_missing:
                missing_counts["any_missing"] += 1

            frame_missing_status["left_hand"].append(left_hand_missing)
            frame_missing_status["right_hand"].append(right_hand_missing)
            frame_missing_status["face"].append(face_missing)
            frame_missing_status["any_missing"].append(any_missing)

        missing_periods = {
            component: self._find_missing_periods(frame_missing_status[component])
            for component in ["left_hand", "right_hand", "face"]
        }

        # interior hand missing - exclude boundary silence
        start, end = self._get_interior_range(
            frame_missing_status["left_hand"],
            frame_missing_status["right_hand"]
        )
        interior_frames = end - start + 1
        interior_hand_missing = sum(
            1 for i in range(start, end + 1)
            if frame_missing_status["left_hand"][i] and frame_missing_status["right_hand"][i]
        )
        interior_hand_missing_pct = (
            interior_hand_missing / interior_frames * 100
        ) if interior_frames > 0 else 0.0

        missing_percentages = {
            component: (count / total_frames * 100) if total_frames > 0 else 0.0
            for component, count in missing_counts.items()
        }

        return {
            "filename": json_file_path.name,
            "total_frames": total_frames,
            "interior_hand_missing_pct": interior_hand_missing_pct,
            "interior_frames": interior_frames,
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
                    "avg_period_length": np.mean([p["length"] for p in periods]) if periods else 0,
                    "max_period_length": max([p["length"] for p in periods]) if periods else 0,
                    "min_period_length": min([p["length"] for p in periods]) if periods else 0
                }
                for component, periods in missing_periods.items()
            }
        }

    def _get_interior_range(self, left_missing: List[bool], right_missing: List[bool]) -> Tuple[int, int]:
        """find first/last frame where any hand is present - boundary silence excluded"""
        n = len(left_missing)
        if n == 0:
            return 0, 0
        start, end = 0, n - 1
        for i in range(n):
            if not left_missing[i] or not right_missing[i]:
                start = i
                break
        for i in range(n - 1, -1, -1):
            if not left_missing[i] or not right_missing[i]:
                end = i
                break
        return start, end

    def _is_hand_data_missing(self, hand_data) -> bool:
        if not hand_data:
            return True
        if isinstance(hand_data, dict):
            landmarks = hand_data.get("landmarks", [])
            confidence = hand_data.get("confidence", 0)
            return not landmarks or confidence < 0.1
        elif isinstance(hand_data, list):
            return len(hand_data) == 0
        return True

    def _is_face_data_missing(self, face_data) -> bool:
        if not face_data:
            return True
        if isinstance(face_data, dict):
            landmarks = face_data.get("all_landmarks", [])
            return not landmarks
        return True

    def _find_missing_periods(self, missing_status_list: List[bool]) -> List[Dict]:
        """find contiguous segments of missing data"""
        periods = []
        in_period = False
        period_start = 0

        for i, is_missing in enumerate(missing_status_list):
            if is_missing and not in_period:
                period_start = i
                in_period = True
            elif not is_missing and in_period:
                periods.append({
                    "start_frame": period_start,
                    "end_frame": i - 1,
                    "length": i - period_start
                })
                in_period = False

        if in_period:
            periods.append({
                "start_frame": period_start,
                "end_frame": len(missing_status_list) - 1,
                "length": len(missing_status_list) - period_start
            })
        return periods

    def _calculate_summary_statistics(
        self,
        sequence_length_data: List[Tuple[str, int]],
        all_missing_percentages: Dict,
        total_missing_periods: int
    ) -> Dict:
        all_lengths = [length for _, length in sequence_length_data]

        summary = {
            "sequence_lengths": {
                "mean": float(np.mean(all_lengths)) if all_lengths else 0,
                "min": int(np.min(all_lengths)) if all_lengths else 0,
                "max": int(np.max(all_lengths)) if all_lengths else 0,
                "median": float(np.median(all_lengths)) if all_lengths else 0,
                "std": float(np.std(all_lengths)) if all_lengths else 0
            },
            "missing_data_percentages": {},
            "total_missing_periods": total_missing_periods,
            "files_with_any_missing": sum(
                1 for p in all_missing_percentages.get("any_missing", []) if p > 0
            )
        }

<<<<<<< HEAD:src/analyze_missing_data.py
        #  Find files with min/max frames
=======
>>>>>>> origin/data-quality:src/AnalyzeMissingData.py
        if sequence_length_data:
            min_len = min(all_lengths)
            max_len = max(all_lengths)
            summary["sequence_lengths"]["min_frames_file"] = next(
                fn for fn, l in sequence_length_data if l == min_len
            )
            summary["sequence_lengths"]["max_frames_file"] = next(
                fn for fn, l in sequence_length_data if l == max_len
            )

        for component, percentages in all_missing_percentages.items():
            if percentages:
                summary["missing_data_percentages"][component] = {
                    "mean": float(np.mean(percentages)),
                    "min": float(np.min(percentages)),
                    "max": float(np.max(percentages)),
                    "median": float(np.median(percentages)),
                    "std": float(np.std(percentages))
                }
            else:
                summary["missing_data_percentages"][component] = {
                    "mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "std": 0.0
                }

        return summary

    def save_analysis(self, analysis_results: Dict):
        with open(self.output_path, "w") as f:
            json.dump(analysis_results, f, indent=4, default=str)
        print(f"Analysis saved to: {self.output_path}")

    def print_analysis_summary(self, analysis_results: Dict):
        metadata = analysis_results["metadata"]
        summary = analysis_results["summary_statistics"]

        print(f"\n{'=' * 50}")
        print("MISSING DATA ANALYSIS SUMMARY")
        print(f"{'=' * 50}")
        print(f"\nDataset Information:")
        print(f"  Directory: {metadata['directory']}")
        print(f"  Files analyzed: {metadata['successful_files']}/{metadata['analyzed_files']}")
        if metadata["failed_files"] > 0:
            print(f"  Failed files: {metadata['failed_files']}")

        print(f"\nSequence Length Statistics:")
<<<<<<< HEAD:src/analyze_missing_data.py
        seq_stats = summary["sequence_lengths"]
        print(f"   Mean length: {seq_stats['mean']:.1f} frames")
        print(f"   Range: {seq_stats['min']:.0f} - {seq_stats['max']:.0f} frames")
        print(f"   Median: {seq_stats['median']:.1f} frames")
        print(f"   Std dev: {seq_stats['std']:.1f} frames")

        #  Show files with min/max frames
        if 'min_frames_file' in seq_stats and 'max_frames_file' in seq_stats:
            print(f"   Shortest sequence: {seq_stats['min_frames_file']} ({seq_stats['min']:.0f} frames)")
            print(f"   Longest sequence: {seq_stats['max_frames_file']} ({seq_stats['max']:.0f} frames)")
=======
        seq = summary["sequence_lengths"]
        print(f"  Mean: {seq['mean']:.1f}  Median: {seq['median']:.1f}  Std: {seq['std']:.1f}")
        print(f"  Range: {seq['min']} - {seq['max']} frames")
        if "min_frames_file" in seq:
            print(f"  Shortest: {seq['min_frames_file']} ({seq['min']} frames)")
            print(f"  Longest:  {seq['max_frames_file']} ({seq['max']} frames)")
>>>>>>> origin/data-quality:src/AnalyzeMissingData.py

        print(f"\nMissing Data Statistics:")
        missing = summary["missing_data_percentages"]
        labels = {
            "left_hand": "Left hand",
            "right_hand": "Right hand",
            "face": "Face",
            "any_missing": "Any missing",
            "interior_hand_missing": "Interior hand missing"
        }
        for key, label in labels.items():
            if key in missing:
                s = missing[key]
                print(f"  {label}: mean={s['mean']:.1f}%  median={s['median']:.1f}%  max={s['max']:.1f}%")

        print(f"\nOverall:")
        print(f"  Total missing periods: {summary['total_missing_periods']}")
        print(f"  Files with any missing: {summary['files_with_any_missing']}/{metadata['successful_files']}")

        self._print_problematic_files(analysis_results)

    def _print_problematic_files(self, analysis_results: Dict):
        print(f"\nFiles with Highest Interior Hand Missing:")
        per_file = analysis_results["per_file_analysis"]
        valid = [
            (fn, d) for fn, d in per_file.items()
            if "interior_hand_missing_pct" in d and "error" not in d
        ]
        if not valid:
            print("  No valid files to show")
            return
        sorted_files = sorted(valid, key=lambda x: x[1]["interior_hand_missing_pct"], reverse=True)
        for i, (fn, d) in enumerate(sorted_files[:10]):
            hand_pct = d["interior_hand_missing_pct"]
            face_pct = d["missing_data_summary"]["face"]["percentage"]
            n = d["total_frames"]
            print(f"  {i + 1:2d}. {fn}  ({n} frames)  hand_interior={hand_pct:.1f}%  face={face_pct:.1f}%")
        if len(sorted_files) > 10:
            print(f"  ... and {len(sorted_files) - 10} more")

    def main(self):
        analysis_results = self.analyze_all_files()
        self.save_analysis(analysis_results)
        self.print_analysis_summary(analysis_results)
        return analysis_results


class AnnotationQualityAnalyzer:
    """
    Annotations-aware quality analyzer.
    Reads a Phoenix-format annotations CSV, matches each row to its raw JSON
    landmark file, computes quality metrics, and can export a filtered CSV.
    """

    def __init__(self, data_dir: str, annotations_path: str, output_path: str = "quality_report.json"):
        self.data_dir = Path(data_dir)
        self.annotations_path = Path(annotations_path)
        self.output_path = Path(output_path)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

        # reuse helper methods without triggering MissingDataAnalyzer's glob scan
        self._helper = _make_file_analyzer_instance(data_dir)
        self.annotations_df = self._load_annotations()
        self._json_index = self._build_json_index()

    def _load_annotations(self) -> pd.DataFrame:
        """load annotations CSV with auto separator detection, same logic as train_lstm.py"""
        suffix = self.annotations_path.suffix.lower()
        if suffix == ".xlsx":
            df = pd.read_excel(self.annotations_path)
        else:
            df = None
            for sep in ["|", ",", "\t", ";"]:
                try:
                    test = pd.read_csv(self.annotations_path, sep=sep, nrows=3)
                    if len(test.columns) >= 4:
                        df = pd.read_csv(self.annotations_path, sep=sep)
                        self._detected_sep = sep
                        break
                except Exception:
                    continue
            if df is None:
                raise ValueError("Could not detect separator in annotations file")

        required = ["id", "folder", "signer", "annotation"]
        if not all(c in df.columns for c in required):
            if len(df.columns) < 4:
                raise ValueError(f"Expected at least 4 columns, got {len(df.columns)}")
            df = df.rename(columns=dict(zip(df.columns[:4], required)))

        df = df[required].copy()
        df = df.dropna()
        df["id"] = df["id"].astype(str).str.strip()
        df["annotation"] = df["annotation"].astype(str).str.strip()
        df = df[df["annotation"].str.len() > 0]
        print(f"Loaded {len(df)} annotation rows from {self.annotations_path}")
        return df

    def _build_json_index(self) -> Dict[str, Path]:
        """index all JSON files by stem for fast lookup"""
        idx = {}
        for p in self.data_dir.rglob("*.json"):
            idx[p.stem] = p
        print(f"Indexed {len(idx)} JSON files under {self.data_dir}")
        return idx

    def _find_json_for_id(self, sample_id: str) -> Optional[Path]:
        """exact stem match first, then substring fallback"""
        if sample_id in self._json_index:
            return self._json_index[sample_id]
        for stem, path in self._json_index.items():
            if sample_id in stem or stem in sample_id:
                return path
        return None

    def analyze(self) -> Dict:
        """
        compute per-sample quality metrics for all annotation rows.
        returns dict keyed by sample id.
        """
        results = {}
        unmatched = 0

        for _, row in self.annotations_df.iterrows():
            sample_id = row["id"]
            json_path = self._find_json_for_id(sample_id)

            if json_path is None:
                print(f"Warning: no JSON file for id={sample_id}")
                unmatched += 1
                results[sample_id] = {"error": "no_file_found", "annotation": row["annotation"]}
                continue

            try:
                file_result = self._helper._analyze_single_json_file(json_path)
                results[sample_id] = {
                    "annotation": row["annotation"],
                    "matched_file": str(json_path),
                    "total_frames": file_result["total_frames"],
                    "interior_hand_missing_pct": file_result["interior_hand_missing_pct"],
                    "face_missing_pct": file_result["missing_data_summary"]["face"]["percentage"],
                    "any_missing_pct": file_result["missing_data_summary"]["any_missing"]["percentage"]
                }
            except Exception as e:
                print(f"Error analyzing {json_path.name} (id={sample_id}): {e}")
                results[sample_id] = {
                    "error": str(e),
                    "annotation": row["annotation"],
                    "matched_file": str(json_path)
                }

        total = len(self.annotations_df)
        matched = total - unmatched
        print(f"Analyzed {matched}/{total} samples ({unmatched} unmatched)")
        return results

    def save_report(self, quality_report: Dict):
        with open(self.output_path, "w") as f:
            json.dump(quality_report, f, indent=4, default=str)
        print(f"Quality report saved to: {self.output_path}")

    def print_report_summary(self, quality_report: Dict):
        valid = [v for v in quality_report.values() if "error" not in v]
        if not valid:
            print("No valid samples in report")
            return

        hand_pcts = [v["interior_hand_missing_pct"] for v in valid]
        face_pcts = [v["face_missing_pct"] for v in valid]

        print(f"\n{'=' * 50}")
        print("ANNOTATION QUALITY REPORT SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Total samples:    {len(quality_report)}")
        print(f"  Analyzable:       {len(valid)}")
        print(f"  Errors/unmatched: {len(quality_report) - len(valid)}")
        print(f"\n  Interior hand missing (%):")
        print(f"    mean={np.mean(hand_pcts):.1f}  median={np.median(hand_pcts):.1f}  max={np.max(hand_pcts):.1f}")
        print(f"\n  Face missing (%):")
        print(f"    mean={np.mean(face_pcts):.1f}  median={np.median(face_pcts):.1f}  max={np.max(face_pcts):.1f}")

        # worst 10 by interior hand missing
        sorted_valid = sorted(valid, key=lambda x: x["interior_hand_missing_pct"], reverse=True)
        print(f"\n  Worst 10 samples (interior hand missing):")
        for entry in sorted_valid[:10]:
            print(
                f"    {Path(entry['matched_file']).stem}"
                f"  hand={entry['interior_hand_missing_pct']:.1f}%"
                f"  face={entry['face_missing_pct']:.1f}%"
                f"  frames={entry['total_frames']}"
            )

    def export_filtered_annotations(
        self,
        quality_report: Dict,
        max_hand_pct: float,
        max_face_pct: float,
        output_path: str,
        min_frames: int = 0
    ):
        """
        write a filtered annotations CSV containing only samples that pass
        all quality thresholds. output is a drop-in replacement for the
        original file (same columns, same separator).
        """
        output_path = Path(output_path)
        passing_ids = set()

        for sample_id, data in quality_report.items():
            if "error" in data:
                continue
            if data["interior_hand_missing_pct"] > max_hand_pct:
                continue
            if data["face_missing_pct"] > max_face_pct:
                continue
            if data["total_frames"] < min_frames:
                continue
            passing_ids.add(sample_id)

        filtered_df = self.annotations_df[self.annotations_df["id"].isin(passing_ids)].copy()

        sep = getattr(self, "_detected_sep", "|")
        filtered_df.to_csv(output_path, sep=sep, index=False)

        total = len(self.annotations_df)
        kept = len(filtered_df)
        dropped = total - kept
        print(f"\nFiltered annotations export:")
        print(f"  Thresholds: hand<={max_hand_pct}%  face<={max_face_pct}%  min_frames>={min_frames}")
        print(f"  Kept:    {kept}/{total}")
        print(f"  Dropped: {dropped}/{total} ({dropped / total * 100:.1f}%)")
        print(f"  Saved to: {output_path}")


def _make_file_analyzer_instance(data_dir: str) -> MissingDataAnalyzer:
    """construct a MissingDataAnalyzer without triggering glob - used internally"""
    obj = object.__new__(MissingDataAnalyzer)
    obj.data_directory = Path(data_dir)
    obj.output_path = Path(".")
    obj.json_files = []
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Analyze missing landmark data in sign language JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing raw JSON landmark files")
    parser.add_argument("--annotations_path", type=str, default=None,
                        help="Path to Phoenix annotations CSV (enables per-sample mode)")
    parser.add_argument("--output", type=str, default="quality_report.json",
                        help="Output JSON report path")
    parser.add_argument("--export_filtered", type=str, default=None,
                        help="If set, export filtered annotations CSV to this path")
    parser.add_argument("--max_hand_pct", type=float, default=30.0,
                        help="Max allowed interior-hand-missing %% (default: 30.0)")
    parser.add_argument("--max_face_pct", type=float, default=50.0,
                        help="Max allowed face-missing %% (default: 50.0)")
    parser.add_argument("--min_frames", type=int, default=0,
                        help="Min sequence length to keep in filtered output (default: 0)")

    args = parser.parse_args()

    if args.annotations_path:
        analyzer = AnnotationQualityAnalyzer(
            data_dir=args.data_dir,
            annotations_path=args.annotations_path,
            output_path=args.output
        )
        report = analyzer.analyze()
        analyzer.save_report(report)
        analyzer.print_report_summary(report)

        if args.export_filtered:
            analyzer.export_filtered_annotations(
                quality_report=report,
                max_hand_pct=args.max_hand_pct,
                max_face_pct=args.max_face_pct,
                output_path=args.export_filtered,
                min_frames=args.min_frames
            )
    else:
        analyzer = MissingDataAnalyzer(args.data_dir, args.output)
        results = analyzer.main()


if __name__ == "__main__":
    main()