#!/usr/bin/env python3

import os
import shutil
import sys
from pathlib import Path


def copy_unlisted_videos():
    """
    Script to copy MP4 videos that DON'T have corresponding JSON filenames in the exclusion list
    """

    # Configuration - Edit these paths as needed
    EXCLUSION_LIST = "H:/Studia/IMU/mgr_backup/mgr/aslcitizen_processed.txt"  # Path to your txt file with .json filenames
    SOURCE_DIR = "H:/Studia/IMU/mgr_backup/mgr/microsoftASLCitizen/ASL_Citizen/ASL_Citizen/videos"  # Directory to check for files
    DEST_DIR = "H:/Studia/IMU/mgr_backup/mgr/microsoftASLCitizen/ASL_Citizen/ASL_Citizen/missed_videos"  # Directory to copy unlisted files to

    # Convert to Path objects for easier handling
    exclusion_list_path = Path(EXCLUSION_LIST)
    source_dir_path = Path(SOURCE_DIR)
    dest_dir_path = Path(DEST_DIR)

    # Check if exclusion list exists
    if not exclusion_list_path.exists():
        print(f"Error: Exclusion list file '{EXCLUSION_LIST}' not found!")
        sys.exit(1)

    # Check if source directory exists
    if not source_dir_path.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found!")
        sys.exit(1)

    # Create destination directory if it doesn't exist
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Checking MP4 files in '{SOURCE_DIR}' against JSON exclusion list '{EXCLUSION_LIST}'")
    print(f"Videos without corresponding JSON entries will be copied to '{DEST_DIR}'")
    print("-" * 70)

    # Read exclusion list and convert .json names to .mp4 names
    try:
        with open(exclusion_list_path, 'r', encoding='utf-8') as f:
            json_filenames = [line.strip() for line in f if line.strip()]

        # Convert .json filenames to .mp4 filenames for comparison
        excluded_video_names = set()
        for json_name in json_filenames:
            if json_name.endswith('.json'):
                # Replace .json with .mp4
                mp4_name = json_name[:-5] + '.mp4'
                excluded_video_names.add(mp4_name)
            else:
                # If it doesn't end with .json, assume it's just the base name
                mp4_name = json_name + '.mp4'
                excluded_video_names.add(mp4_name)

    except Exception as e:
        print(f"Error reading exclusion list: {e}")
        sys.exit(1)

    print(f"Loaded {len(json_filenames)} JSON filenames from exclusion list")
    print(f"Converted to {len(excluded_video_names)} MP4 filenames to exclude")
    print("-" * 70)

    # Show some examples of what will be excluded
    if excluded_video_names:
        print("Examples of MP4 files that will be SKIPPED (first 5):")
        for i, name in enumerate(sorted(excluded_video_names)):
            if i >= 5:
                break
            print(f"  - {name}")
        print("-" * 70)

    # Counters
    copied_count = 0
    total_count = 0

    # Find all .mp4 files in source directory
    mp4_files = list(source_dir_path.glob("*.mp4"))

    # Debug: Show what we found
    print(f"Found {len(mp4_files)} .mp4 files in source directory")

    if not mp4_files:
        print("No .mp4 files found in source directory")
        print(f"Checked directory: {source_dir_path.absolute()}")
        print("Please verify:")
        print("1. The source directory path is correct")
        print("2. The directory contains .mp4 files")
        return

    # Process each MP4 file
    for file_path in mp4_files:
        filename = file_path.name
        total_count += 1

        # Check if this MP4 filename corresponds to a JSON in the exclusion list
        if filename in excluded_video_names:
            print(f"SKIP: {filename} (corresponding JSON found in exclusion list)")
        else:
            print(f"COPY: {filename} (no corresponding JSON in exclusion list)")
            try:
                # Copy file to destination
                dest_file = dest_dir_path / filename
                shutil.copy2(file_path, dest_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")

    print("-" * 70)
    print("Summary:")
    print(f"Total .mp4 files processed: {total_count}")
    print(f"Videos copied: {copied_count}")
    print(f"Videos skipped: {total_count - copied_count}")
    print(f"Copied videos are in: {DEST_DIR}")


def main():
    """Main function with optional command line arguments"""

    # You can optionally accept command line arguments
    if len(sys.argv) == 4:
        copy_unlisted_videos_with_args(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 1:
        # Use default values from the function
        copy_unlisted_videos()
    else:
        print("Usage:")
        print("  python copy_unlisted_videos.py")
        print("  python copy_unlisted_videos.py <json_exclusion_list> <source_dir> <dest_dir>")
        print("")
        print("This script copies MP4 videos that DON'T have corresponding JSON filenames")
        print("in the exclusion list.")
        sys.exit(1)


def copy_unlisted_videos_with_args(exclusion_list, source_dir, dest_dir):
    """Version that accepts arguments"""

    # Convert to Path objects
    exclusion_list_path = Path(exclusion_list)
    source_dir_path = Path(source_dir)
    dest_dir_path = Path(dest_dir)

    # Check if exclusion list exists
    if not exclusion_list_path.exists():
        print(f"Error: Exclusion list file '{exclusion_list}' not found!")
        sys.exit(1)

    # Check if source directory exists
    if not source_dir_path.exists():
        print(f"Error: Source directory '{source_dir}' not found!")
        sys.exit(1)

    # Create destination directory if it doesn't exist
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Checking MP4 files in '{source_dir}' against JSON exclusion list '{exclusion_list}'")
    print(f"Videos without corresponding JSON entries will be copied to '{dest_dir}'")
    print("-" * 70)

    # Read exclusion list and convert .json names to .mp4 names
    try:
        with open(exclusion_list_path, 'r', encoding='utf-8') as f:
            json_filenames = [line.strip() for line in f if line.strip()]

        # Convert .json filenames to .mp4 filenames for comparison
        excluded_video_names = set()
        for json_name in json_filenames:
            if json_name.endswith('.json'):
                # Replace .json with .mp4
                mp4_name = json_name[:-5] + '.mp4'
                excluded_video_names.add(mp4_name)
            else:
                # If it doesn't end with .json, assume it's just the base name
                mp4_name = json_name + '.mp4'
                excluded_video_names.add(mp4_name)

    except Exception as e:
        print(f"Error reading exclusion list: {e}")
        sys.exit(1)

    print(f"Loaded {len(json_filenames)} JSON filenames from exclusion list")
    print(f"Converted to {len(excluded_video_names)} MP4 filenames to exclude")
    print("-" * 70)

    # Counters
    copied_count = 0
    total_count = 0

    # Find all .mp4 files in source directory
    mp4_files = list(source_dir_path.glob("*.mp4"))

    if not mp4_files:
        print("No .mp4 files found in source directory")
        return

    # Process each MP4 file
    for file_path in mp4_files:
        filename = file_path.name
        total_count += 1

        # Check if this MP4 filename corresponds to a JSON in the exclusion list
        if filename in excluded_video_names:
            print(f"SKIP: {filename} (corresponding JSON found in exclusion list)")
        else:
            print(f"COPY: {filename} (no corresponding JSON in exclusion list)")
            try:
                # Copy file to destination
                dest_file = dest_dir_path / filename
                shutil.copy2(file_path, dest_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")

    print("-" * 70)
    print("Summary:")
    print(f"Total .mp4 files processed: {total_count}")
    print(f"Videos copied: {copied_count}")
    print(f"Videos skipped: {total_count - copied_count}")
    print(f"Copied videos are in: {dest_dir}")


if __name__ == "__main__":
    main()