import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
from pathlib import Path
from processing import process_video

TOP_N = 20  # Easy to modify: 1 for testing, 20 for final processing


def analyze_gloss_frequencies(csv_file_path, output_json_path="gloss_frequencies.json",
                              histogram_path="gloss_histogram.png", top_glosses_to_save=50):
    """
    Analyze gloss frequencies in the 'Gloss' column of a CSV file.
    Treats each entire gloss entry as a single unit (e.g., "SPECIAL TODAY" is one unit, not two words).

    Args:
        csv_file_path (str): Path to the input CSV file
        output_json_path (str): Path to save the frequency data as JSON
        histogram_path (str): Path to save the histogram image
        top_glosses_to_save (int): Number of top glosses to save in results (default: 50)

    Returns:
        dict: Dictionary containing gloss frequency statistics
    """

    try:
        # Read the CSV file
        print(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)

        # Check if 'Gloss' column exists
        if 'Gloss' not in df.columns:
            raise ValueError(f"'Gloss' column not found. Available columns: {list(df.columns)}")

        print(f"Found {len(df)} rows in the dataset")
        print(f"Columns: {list(df.columns)}")

        # Extract and clean the Gloss column - treat each gloss as a single unit
        gloss_data = df['Gloss'].dropna()  # Remove NaN values
        print(f"Non-null Gloss entries: {len(gloss_data)}")

        # Collect all gloss entries as single units (no splitting)
        all_glosses = []
        for gloss_entry in gloss_data:
            # Convert to string and clean
            gloss_str = str(gloss_entry).strip()

            # Clean: remove extra whitespace and convert to uppercase
            clean_gloss = ' '.join(gloss_str.split()).upper()  # Normalize whitespace

            if clean_gloss:  # Only add non-empty glosses
                all_glosses.append(clean_gloss)

        print(f"Total gloss entries extracted: {len(all_glosses)}")

        # Count gloss frequencies (treating each entire gloss as one unit)
        gloss_frequencies = Counter(all_glosses)
        unique_gloss_count = len(gloss_frequencies)

        print(f"Unique glosses found: {unique_gloss_count}")

        # Get the most common glosses first (this maintains proper order)
        most_common_list = gloss_frequencies.most_common(top_glosses_to_save)

        # Create ordered dictionary from most_common to ensure alignment
        ordered_frequencies = {gloss: count for gloss, count in most_common_list}

        # Prepare statistics
        stats = {
            "total_glosses": len(all_glosses),
            "unique_glosses": unique_gloss_count,
            "gloss_frequencies": ordered_frequencies,  # Now properly aligned
            "most_common_glosses": most_common_list,
            "analysis_metadata": {
                "csv_file": str(csv_file_path),
                "total_rows": len(df),
                "non_null_glosses": len(gloss_data)
            }
        }

        # Save to JSON file
        print(f"Saving frequency data to: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # Create histogram
        create_frequency_histogram(gloss_frequencies, histogram_path, stats)

        # Print summary
        print_analysis_summary(stats)

        return stats

    except Exception as e:
        print(f"Error analyzing gloss frequencies: {e}")
        raise


def create_frequency_histogram(word_frequencies, histogram_path, stats):
    """
    Create and save a histogram of word frequencies.

    Args:
        word_frequencies (Counter): Counter object with word frequencies
        histogram_path (str): Path to save the histogram
        stats (dict): Statistics dictionary for title information
    """

    try:
        # Get frequency values
        frequencies = list(word_frequencies.values())

        if not frequencies:
            print("No frequency data to plot")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Frequency distribution histogram
        ax1.hist(frequencies, bins=min(50, len(set(frequencies))),
                 alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Gloss Frequency')
        ax1.set_ylabel('Number of Glosses')
        ax1.set_title('Distribution of Gloss Frequencies')
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        mean_freq = np.mean(frequencies)
        median_freq = np.median(frequencies)
        max_freq = max(frequencies)

        stats_text = f'Mean: {mean_freq:.1f}\nMedian: {median_freq:.1f}\nMax: {max_freq}'
        ax1.text(0.7, 0.9, stats_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        # Plot 2: Top N most frequent glosses
        most_common = word_frequencies.most_common(TOP_N)
        if most_common:
            glosses, counts = zip(*most_common)

            y_pos = np.arange(len(glosses))
            ax2.barh(y_pos, counts, color='lightcoral')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(glosses)
            ax2.set_xlabel('Frequency')
            ax2.set_title(f'Top {TOP_N} Most Frequent Glosses')
            ax2.grid(True, alpha=0.3, axis='x')

            # Invert y-axis to show highest frequency at top
            ax2.invert_yaxis()

        # Overall title
        fig.suptitle(f'Gloss Frequency Analysis\n'
                     f'Total Glosses: {stats["total_glosses"]}, '
                     f'Unique Glosses: {stats["unique_glosses"]}',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save the plot
        print(f"Saving histogram to: {histogram_path}")
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close()

        print("Histogram created successfully")

    except Exception as e:
        print(f"Error creating histogram: {e}")


def print_analysis_summary(stats):
    """Print a summary of the analysis results."""

    print("\n" + "=" * 50)
    print("GLOSS FREQUENCY ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"Total gloss entries processed: {stats['total_glosses']}")
    print(f"Unique glosses found: {stats['unique_glosses']}")
    print(f"Average frequency per unique gloss: {stats['total_glosses'] / stats['unique_glosses']:.2f}")

    print(f"\nDataset Information:")
    print(f"  - CSV file: {stats['analysis_metadata']['csv_file']}")
    print(f"  - Total rows: {stats['analysis_metadata']['total_rows']}")
    print(f"  - Non-null glosses: {stats['analysis_metadata']['non_null_glosses']}")

    print(f"\nTop 10 Most Frequent Glosses:")
    for i, (gloss, count) in enumerate(stats['most_common_glosses'][:10], 1):
        percentage = (count / stats['total_glosses']) * 100
        print(f"  {i:2d}. {gloss:<20} : {count:4d} times ({percentage:5.1f}%)")

    print("=" * 50)


def validate_csv_structure(csv_file_path):
    """
    Validate that the CSV file has the expected structure.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        bool: True if valid, False otherwise
    """

    try:
        df = pd.read_csv(csv_file_path, nrows=5)  # Read only first 5 rows for validation

        required_columns = ['Participant ID', 'Video file', 'Gloss', 'ASL-LEX Code']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing expected columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")

            if 'Gloss' not in df.columns:
                print("Error: 'Gloss' column is required but not found!")
                return False

        # Show sample of Gloss data to help user understand the format
        if 'Gloss' in df.columns:
            print(f"\nSample Gloss entries:")
            for i, gloss in enumerate(df['Gloss'].dropna().head(3)):
                print(f"  Row {i + 1}: '{gloss}'")

        print("CSV structure validation passed")
        return True

    except Exception as e:
        print(f"Error validating CSV structure: {e}")
        return False


def filter_data_for_top_words(csv_file_path, output_csv_path=f"gloss_{TOP_N}.csv", top_n=TOP_N):
    """
    Filter CSV data to include only rows where video filenames contain words from the top N most common glosses.

    Args:
        csv_file_path (str): Path to the input CSV file
        output_csv_path (str): Path to save the filtered CSV file
        top_n (int): Number of top glosses to filter for (default: 20)

    Returns:
        tuple: (filtered_dataframe, top_glosses_list, statistics)
    """

    try:
        print(f"\n=== Filtering Data for Top {top_n} Glosses ===")

        # First, analyze frequencies to get top glosses
        print("Step 1: Analyzing gloss frequencies...")
        stats = analyze_gloss_frequencies(csv_file_path,
                                          output_json_path="temp_frequencies.json",
                                          histogram_path="temp_histogram.png",
                                          top_glosses_to_save=max(50, top_n * 2))  # Ensure we have enough glosses

        # Get top N glosses (entire gloss entries, not individual words)
        top_glosses = [gloss for gloss, count in stats['most_common_glosses'][:top_n]]
        print(f"\nTop {top_n} most common glosses:")
        for i, (gloss, count) in enumerate(stats['most_common_glosses'][:top_n], 1):
            print(f"  {i:2d}. {gloss:<20} ({count} times)")

        # Load the full CSV
        print(f"\nStep 2: Loading full CSV data...")
        df = pd.read_csv(csv_file_path)
        print(f"Total rows in original data: {len(df)}")

        # Filter rows where the Gloss column exactly matches one of the top glosses
        print(f"\nStep 3: Filtering data...")
        filtered_rows = []

        # Convert top glosses to uppercase for comparison
        top_glosses_upper = [gloss.upper() for gloss in top_glosses]

        print(f"Will match rows where Gloss column exactly matches these top {top_n} glosses:")
        for i, gloss in enumerate(top_glosses, 1):
            print(f"  {i:2d}. '{gloss}'")

        # Debug: Track exact gloss matches
        gloss_match_counts = {}

        for idx, row in df.iterrows():
            gloss_content = str(row['Gloss']).strip().upper()

            # Check if this row's gloss exactly matches one of the top glosses
            if gloss_content in top_glosses_upper:
                filtered_rows.append(row)

                # Track the matches for debugging
                if gloss_content not in gloss_match_counts:
                    gloss_match_counts[gloss_content] = 0
                gloss_match_counts[gloss_content] += 1

        # Debug output
        print(f"\n=== FILTERING DEBUG INFORMATION ===")
        print(f"Exact gloss matches found:")
        sorted_gloss_matches = sorted(gloss_match_counts.items(), key=lambda x: x[1], reverse=True)
        for gloss, count in sorted_gloss_matches:
            print(f"  - '{gloss}': {count} videos")

        # Show comparison with original top glosses
        print(f"\nComparison with original frequency analysis:")
        for i, (gloss, original_count) in enumerate(stats['most_common_glosses'][:top_n], 1):
            filtered_count = gloss_match_counts.get(gloss.upper(), 0)
            print(f"  {i}. '{gloss}': {original_count} in analysis → {filtered_count} videos found")
            if filtered_count != original_count:
                print(f"       MISMATCH: Expected {original_count}, found {filtered_count} videos")
            else:
                print(f"      PERFECT MATCH!")

        print(f"=== END DEBUG INFORMATION ===\n")

        # Create filtered dataframe
        if filtered_rows:
            filtered_df = pd.DataFrame(filtered_rows)
            filtered_df = filtered_df.reset_index(drop=True)
        else:
            filtered_df = pd.DataFrame(columns=df.columns)

        print(f"Filtered rows: {len(filtered_df)}")

        # Save filtered data
        print(f"\nStep 4: Saving filtered data to {output_csv_path}")
        filtered_df.to_csv(output_csv_path, index=False)

        # Create statistics
        filter_stats = {
            "original_rows": len(df),
            "filtered_rows": len(filtered_df),
            "top_glosses_used": top_glosses,
            "top_n": top_n,
            "retention_rate": len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
        }

        # Show some examples of filtered data
        if len(filtered_df) > 0:
            print(f"\nSample filtered entries:")
            for i in range(min(5, len(filtered_df))):
                row = filtered_df.iloc[i]
                print(f"  {i + 1}. Video: {row['Video file']}")
                print(f"     Gloss: {row['Gloss']}")
                print()

        print(f"Filter Statistics:")
        print(f"  - Original rows: {filter_stats['original_rows']}")
        print(f"  - Filtered rows: {filter_stats['filtered_rows']}")
        print(f"  - Retention rate: {filter_stats['retention_rate']:.1f}%")

        # Clean up temporary files
        try:
            Path("temp_frequencies.json").unlink(missing_ok=True)
            Path("temp_histogram.png").unlink(missing_ok=True)
        except:
            pass

        return filtered_df, top_glosses, filter_stats

    except Exception as e:
        print(f"Error filtering data: {e}")
        raise


def process_filtered_videos(filtered_df, video_dir, output_dir, args=None):
    """
    Process the actual video files mentioned in the filtered dataset using your existing process_video function.
    For ASL-20 processing, outputs only JSON files with landmarks using video filenames.

    Args:
        filtered_df (pd.DataFrame): Filtered dataframe containing video filenames
        video_dir (str): Directory containing video files
        output_dir (str): Directory to save processed outputs
        args: Processing arguments (optional)

    Returns:
        dict: Video processing statistics
    """
    try:
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)

        print(f"DEBUG: Video directory: {video_dir}")
        print(f"DEBUG: Output directory: {output_dir}")
        print(f"DEBUG: Video directory exists: {video_dir.exists()}")

        # Validate video directory
        if not video_dir.exists():
            raise ValueError(f"Video directory does not exist: {video_dir}")

        # Create subdirectory for video processing outputs (similar to --process-phoenix)
        video_output_dir = output_dir / f"asl{TOP_N}_landmarks"
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"DEBUG: Video output directory: {video_output_dir}")

        # Get list of video files to process
        video_files_to_process = []
        missing_files = []

        print(f"DEBUG: Checking {len(filtered_df)} filtered rows for video files...")

        for _, row in filtered_df.iterrows():
            video_filename = str(row['Video file'])
            video_path = video_dir / video_filename  # Use video_dir, NOT output_dir

            print(f"DEBUG: Looking for video: {video_path}")

            if video_path.exists():
                video_files_to_process.append({
                    'path': video_path,
                    'filename': video_filename,
                    'gloss': row['Gloss'],
                    'participant_id': row['Participant ID'],
                    'asl_lex_code': row.get('ASL-LEX Code', 'N/A')
                })
                print(f"DEBUG: ✓ Found video: {video_filename}")
            else:
                missing_files.append(video_filename)
                print(f"DEBUG: ✗ Missing video: {video_filename}")

        print(f"Found {len(video_files_to_process)} videos to process")
        if missing_files:
            print(f"Warning: {len(missing_files)} video files not found in {video_dir}")
            print(f"First few missing: {missing_files[:5]}")

        # Process videos using your existing process_video function
        successful_count = 0
        failed_count = 0

        print(f"\n{'=' * 60}")
        print(f"STARTING VIDEO PROCESSING - JSON ONLY MODE")
        print(f"{'=' * 60}")
        print(f"Total videos to process: {len(video_files_to_process)}")
        print(f"{'=' * 60}")

        for i, video_info in enumerate(video_files_to_process):
            try:
                # Enhanced progress logging
                progress = f"[{i + 1:3d}/{len(video_files_to_process):3d}]"
                print(f"\n{progress} Processing: {video_info['filename']}")
                print(f"        Progress: {((i + 1) / len(video_files_to_process) * 100):5.1f}% complete")
                print(f"        Input: {video_info['path']}")

                # Extract filename without extension for JSON naming
                video_stem = Path(video_info['filename']).stem
                print(f"        Output JSON: {video_stem}.json")

                # Use your actual process_video function with JSON-only mode
                result = process_video(
                    input_path=str(video_info['path']),  # Full path to video file
                    output_dir=str(video_output_dir),  # Output directory for landmarks
                    skip_frames=getattr(args, 'skip_frames', 1) if args else 1,
                    extract_face=True,
                    extract_pose=True,
                    is_image_sequence=False,
                    save_all_frames=False,  # Force to False for JSON-only mode
                    use_full_mesh=getattr(args, 'full_mesh', False) if args else False,
                    use_enhancement=getattr(args, 'enhance', False) if args else False,
                    phoenix_mode=True,  # Enable Phoenix mode for JSON-only processing
                    phoenix_json_only=True,  # JSON-only mode (no frames/videos)
                    phoenix_json_name=video_stem,  # Custom JSON filename (without extension)
                    disable_mirroring=getattr(args, 'disable_mirroring', False) if args else False,
                    args=args
                )

                if result is not None:
                    successful_count += 1
                    print(f"        ✓ SUCCESS: {video_stem}.json created")
                else:
                    print(f"        ✗ FAILED: No result returned")
                    failed_count += 1

                # Show running totals every 10 videos or on important milestones
                if (i + 1) % 10 == 0 or (i + 1) in [1, 5, 25, 50, 100]:
                    print(f"        📊 Running totals: {successful_count} success, {failed_count} failed")

            except Exception as e:
                print(f"        ✗ ERROR: {e}")
                failed_count += 1

        # Create processing statistics
        processing_stats = {
            "total_videos_in_csv": len(filtered_df),
            "videos_found": len(video_files_to_process),
            "videos_missing": len(missing_files),
            "successful": successful_count,
            "failed": failed_count,
            "output_directory": str(video_output_dir),
            "video_directory": str(video_dir),  # Add this for debugging
            "missing_files": missing_files[:10] if missing_files else [],
            "processing_mode": "JSON_ONLY",  # Indicate the processing mode
            "processing_settings": {
                "skip_frames": getattr(args, 'skip_frames', 1) if args else 1,
                "full_mesh": getattr(args, 'full_mesh', False) if args else False,
                "enhance": getattr(args, 'enhance', False) if args else False,
                "phoenix_json_only": True,
                "save_all_frames": False
            }
        }

        print(f"\n{'=' * 60}")
        print(f"VIDEO PROCESSING COMPLETE!")
        print(f"{'=' * 60}")
        print(f"FINAL RESULTS:")
        print(f"  - Total videos processed: {len(video_files_to_process)}")
        print(f"  - Successful: {successful_count} ")
        print(f"  - Failed: {failed_count} ")
        print(
            f"  - Success rate: {(successful_count / len(video_files_to_process) * 100):5.1f}%" if video_files_to_process else "N/A")
        print(f"")
        print(f"OUTPUT DETAILS:")
        print(f"  - Video directory: {video_dir}")
        print(f"  - Output directory: {video_output_dir}")
        print(f"  - Processing mode: JSON landmarks only")
        print(f"  - Videos in filtered CSV: {processing_stats['total_videos_in_csv']}")
        print(f"  - Videos found: {processing_stats['videos_found']}")
        print(f"  - Videos missing: {processing_stats['videos_missing']}")
        if missing_files:
            print(f"  - First few missing files: {missing_files[:3]}")

        # List the JSON files created
        json_files = list(video_output_dir.glob("*.json"))
        print(f"")
        print(f"JSON FILES CREATED:")
        print(f"  - Total JSON files: {len(json_files)}")
        if json_files:
            print(f"  - Sample JSON files: {[f.name for f in json_files[:5]]}")
        print(f"{'=' * 60}")

        return processing_stats

    except Exception as e:
        print(f"Error processing videos: {e}")
        return {"error": str(e)}


def process_microsoftasl_20(csv_file_path, output_dir="./", video_dir=None, process_videos=False, args=None):
    """
    Main function for --process-microsoftasl-20 flag.
    Analyzes the CSV file and creates a filtered dataset with the 20 most common glosses.
    Optionally processes the actual video files mentioned in the filtered dataset.

    Args:
        csv_file_path (str): Path to the input CSV file
        output_dir (str): Directory to save output files
        video_dir (str): Directory containing video files (optional)
        process_videos (bool): Whether to process the actual video files
        args: Processing arguments (optional)

    Returns:
        dict: Processing results and statistics
    """

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"MICROSOFT ASL-{TOP_N} DATASET PROCESSING")
        print(f"{'=' * 60}")
        print(f"Input CSV: {csv_file_path}")
        print(f"Output directory: {output_dir}")

        # Add debugging information
        if process_videos:
            print(f"Video processing: ENABLED")
            print(f"Video directory: {video_dir}")
            if video_dir:
                video_dir_path = Path(video_dir)
                print(f"Video directory exists: {video_dir_path.exists()}")
                if video_dir_path.exists():
                    video_files = list(video_dir_path.glob("*.mp4"))
                    print(f"Found {len(video_files)} .mp4 files in video directory")
                    if video_files:
                        print(f"Sample video files: {[f.name for f in video_files[:3]]}")
            else:
                print("ERROR: video_dir is None!")
                return {"error": "video_dir parameter is None"}
        else:
            print(f"Video processing: DISABLED")

        # Validate CSV structure
        if not validate_csv_structure(csv_file_path):
            raise ValueError("CSV validation failed")

        # Create output paths
        filtered_csv_path = output_dir / f"gloss_{TOP_N}.csv"
        frequencies_json_path = output_dir / f"gloss_{TOP_N}_frequencies.json"
        histogram_path = output_dir / f"gloss_{TOP_N}_histogram.png"

        # Filter data for top N glosses
        filtered_df, top_glosses, filter_stats = filter_data_for_top_words(
            csv_file_path=csv_file_path,
            output_csv_path=str(filtered_csv_path),
            top_n=TOP_N
        )

        # Generate analysis for the filtered data
        if len(filtered_df) > 0:
            print(f"\nGenerating analysis for filtered data...")
            analyze_gloss_frequencies(
                csv_file_path=str(filtered_csv_path),
                output_json_path=str(frequencies_json_path),
                histogram_path=str(histogram_path),
                top_glosses_to_save=TOP_N  # Save enough glosses for analysis
            )

        # Process actual video files if requested
        video_processing_stats = None
        if process_videos and video_dir and len(filtered_df) > 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING VIDEO FILES")
            print(f"{'=' * 60}")

            # Additional validation before processing videos
            video_dir_path = Path(video_dir)
            if not video_dir_path.exists():
                print(f"ERROR: Video directory does not exist: {video_dir_path}")
                return {"error": f"Video directory does not exist: {video_dir_path}"}

            video_processing_stats = process_filtered_videos(filtered_df, video_dir, output_dir, args)
        elif process_videos and not video_dir:
            print(f"ERROR: Video processing requested but video_dir is not provided!")
            return {"error": "Video processing requested but video_dir is not provided"}

        # Create summary report
        summary = {
            "processing_type": f"Microsoft ASL-{TOP_N} Dataset",
            "input_file": str(csv_file_path),
            "output_files": {
                "filtered_csv": str(filtered_csv_path),
                "frequencies_json": str(frequencies_json_path),
                "histogram": str(histogram_path)
            },
            "statistics": filter_stats,
            f"top_{TOP_N}_glosses": top_glosses,
            "video_processing": video_processing_stats
        }

        # Save summary
        summary_path = output_dir / f"gloss_{TOP_N}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'=' * 60}")
        print(f"Files created:")
        print(f"  - Filtered data: {filtered_csv_path}")
        print(f"  - Gloss frequencies: {frequencies_json_path}")
        print(f"  - Histogram: {histogram_path}")
        print(f"  - Summary report: {summary_path}")
        if video_processing_stats and 'error' not in video_processing_stats:
            print(f"  - Video landmarks: {video_processing_stats['output_directory']}")
            print(
                f"  - Videos processed: {video_processing_stats['successful']}/{video_processing_stats['videos_found']}")

        print(f"\nTop {TOP_N} glosses: {', '.join(top_glosses)}")
        print(
            f"Dataset reduced from {filter_stats['original_rows']} to {filter_stats['filtered_rows']} rows ({filter_stats['retention_rate']:.1f}%)")

        return summary

    except Exception as e:
        print(f"Error in Microsoft ASL-20 processing: {e}")
        raise


# Example usage function
def main():
    """
    Standalone usage: Analyze CSV file and produce complete analysis with all 4 output files.
    """

    # Specify your CSV file path here
    csv_file_path = "your_data.csv"  # Change this to your actual file path

    # Check if CSV file exists
    if not Path(csv_file_path).exists():
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please update the csv_file_path variable with your actual file path")
        return

    # Run the complete Microsoft ASL-N analysis (same as command line option)
    try:
        print(f"Running complete Microsoft ASL-{TOP_N} analysis...")
        results = process_microsoftasl_20(csv_file_path, output_dir="./")

        print(f"\nStandalone analysis completed successfully!")
        print(f"Created {len(results['output_files'])} output files")

    except Exception as e:
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()