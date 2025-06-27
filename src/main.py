#!/usr/bin/env python3
import argparse
import cv2
import json
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Any
from processing import process_image, process_video, enhance_image_for_hand_detection

# MediaPipe solution instances
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Mouth landmark indices from MediaPipe FaceMesh
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]

# Default settings
DEFAULT_OUTPUT_DIR = Path("./../output")
DEFAULT_DATA_DIR = Path("./../data")
VISUALIZE_ENHANCEMENTS = False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process images and videos for sign language recognition')

    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save output files')

    # Image processing options
    parser.add_argument('--images', type=str, nargs='+', default=[],
                        help='List of image files to process')
    parser.add_argument('--skip-image-processing', action='store_true',
                        help='Skip processing of individual images')
    parser.add_argument('--visualize-enhancements', action='store_true',
                        help='Add additional visual enhancements to output images')

    # Video processing options
    parser.add_argument('--videos', type=str, nargs='+', default=[],
                        help='List of video files to process')
    parser.add_argument('--image-dirs', type=str, nargs='+', default=DEFAULT_DATA_DIR,
                        help='List of directories containing image sequences to process as videos')
    parser.add_argument('--image-extension', type=str, default='png',
                        help='File extension for image sequences (jpg, png, etc.)')
    parser.add_argument('--skip-video-processing', action='store_true',
                        help='Skip processing of videos')
    parser.add_argument('--skip-frames', type=int, default=2,
                        help='Process every nth frame in videos')
    parser.add_argument('--save-all-frames', action='store_true',
                        help='Save all annotated frames to disk')

    # Detection options
    parser.add_argument('--detect-faces', action='store_true',
                        help='Enable face landmark detection')
    parser.add_argument('--detect-pose', action='store_true',
                        help='Enable pose landmark detection')

    return parser.parse_args()


def process_images(image_files: List[str], output_dir: Path, detect_faces: bool,
                   detect_pose: bool, visualize_enhancements: bool) -> Dict[str, Any]:
    """
    Process multiple image files and extract landmarks.

    Args:
        image_files: List of image file paths
        output_dir: Directory to save output files
        detect_faces: Whether to detect face landmarks
        detect_pose: Whether to detect pose landmarks
        visualize_enhancements: Whether to visualize image enhancement steps

    Returns:
        Dictionary of landmarks data for all images
    """
    landmarks_data = {}

    for idx, file_path in enumerate(image_files):
        file_path = Path(file_path)
        print(f"Processing image [{idx + 1}/{len(image_files)}]: {file_path}")

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        # Load and enhance the image
        original_image = cv2.imread(str(file_path))
        if original_image is None:
            print(f"Warning: Could not read image: {file_path}")
            continue

        enhanced_image = enhance_image_for_hand_detection(
            original_image,
            visualize=visualize_enhancements
        )

        # Use the enhanced image in process_image
        image_data, annotated_image = process_image(
            enhanced_image,
            detect_faces=detect_faces,
            detect_pose=detect_pose
        )

        if image_data and annotated_image is not None:
            landmarks_data[str(file_path)] = image_data

            output_filename = f"annotated_{file_path.stem}.png"
            output_path = output_dir / output_filename

            cv2.imwrite(str(output_path), annotated_image)
            print(f"Annotated image saved to {output_path}")

    return landmarks_data


def get_subdirectories(directory_path: str) -> List[str]:
    """
    Get all immediate subdirectories of the given directory.

    Args:
        directory_path: Path to the directory

    Returns:
        List of subdirectory paths
    """
    base_path = Path(directory_path)

    if not base_path.exists() or not base_path.is_dir():
        print(f"Warning: Directory not found or is not a directory: {directory_path}")
        return []

    # Get all immediate subdirectories
    subdirectories = [str(item) for item in base_path.iterdir() if item.is_dir()]

    if not subdirectories:
        print(f"No subdirectories found in: {directory_path}")

    return subdirectories


def process_videos(
        input_paths: List[str],
        output_dir: Path,
        skip_frames: int,
        extract_face: bool,
        extract_pose: bool,
        input_types: List[bool],
        image_extension: str = "png",
        save_all_frames: bool = False  # Add the new parameter
) -> None:
    """
    Process multiple video files or image sequence directories and extract landmarks.

    Args:
        input_paths: List of video file paths or image directory paths
        output_dir: Directory to save output files
        skip_frames: Process every nth frame
        extract_face: Whether to extract face landmarks
        extract_pose: Whether to extract pose landmarks
        input_types: List of booleans indicating if each input is an image sequence (True) or video (False)
        image_extension: File extension to look for when processing image sequences
        save_all_frames: Whether to save all annotated frames to disk
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created/verified: {output_dir.absolute()}")
    except Exception as e:
        print(f"ERROR: Failed to create output directory {output_dir}: {str(e)}")
        return

    if not input_paths:
        print("No input paths provided for video/image sequence processing")
        return

    # Process each input
    for idx, (input_path, is_image_sequence) in enumerate(zip(input_paths, input_types)):
        input_path = Path(input_path)
        input_type = "image directory" if is_image_sequence else "video"

        print(f"Processing {input_type} [{idx + 1}/{len(input_paths)}]: {input_path}")

        if not input_path.exists():
            print(f"WARNING: {input_type} not found: {input_path}")
            continue

        # Create output subdirectory
        if is_image_sequence:
            output_subdir = output_dir / f"images_{input_path.name}"
        else:
            output_subdir = output_dir / f"video_{input_path.stem}"

        try:
            output_subdir.mkdir(parents=True, exist_ok=True)
            print(f"Created output subdirectory: {output_subdir.absolute()}")
        except Exception as e:
            print(f"ERROR: Failed to create output subdirectory {output_subdir}: {str(e)}")
            continue

        # Process video or image sequence
        print(f"Starting processing of {input_type} to output directory: {output_subdir}")
        try:
            result =   process_video(
                str(input_path),
                output_dir=str(output_subdir),
                skip_frames=skip_frames,
                extract_face=extract_face,
                extract_pose=extract_pose,
                is_image_sequence=is_image_sequence,
                image_extension=image_extension,
                save_all_frames=save_all_frames
            )

            if result is None:
                print(f"WARNING: Processing {input_type} {input_path} returned None. Check for errors.")
            else:
                print(f"Successfully processed {input_type} {input_path} with {len(result)} frames of data.")

            # Verify output files exist
            expected_video_file = Path(output_subdir) / "annotated_video.mp4"
            expected_json_file = Path(output_subdir) / "video_landmarks.json"

            if expected_video_file.exists():
                print(f"Confirmed output video file exists: {expected_video_file}")
                print(f"Video file size: {expected_video_file.stat().st_size} bytes")
            else:
                print(f"ERROR: Output video file not found: {expected_video_file}")

            if expected_json_file.exists():
                print(f"Confirmed output JSON file exists: {expected_json_file}")
                print(f"JSON file size: {expected_json_file.stat().st_size} bytes")
            else:
                print(f"ERROR: Output JSON file not found: {expected_json_file}")

        except Exception as e:
            print(f"ERROR: Failed to process {input_type} {input_path}: {str(e)}")
            import traceback
            traceback.print_exc()


def main() -> None:
    """Main function to process images, videos, or image sequences"""
    # Parse command line arguments
    args = parse_arguments()

    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tmp").mkdir(exist_ok=True)

    landmarks_data = {}

    # Process static images
    if not args.skip_image_processing and args.images:
        print("\n=== Processing Individual Images ===")
        landmarks_data = process_images(
            args.images,
            output_dir,
            args.detect_faces,
            args.detect_pose,
            args.visualize_enhancements
        )

        # Save all landmark data to JSON
        if landmarks_data:
            json_output_path = output_dir / "landmarks_data.json"
            with open(json_output_path, "w") as json_file:
                json.dump(landmarks_data, json_file, indent=4)

            print(f"Landmark data saved to {json_output_path}")

    # Process videos and image sequences
    if not args.skip_video_processing:
        all_inputs = []
        input_types = []

        # Add video files
        if args.videos:
            all_inputs.extend(args.videos)
            input_types.extend([False] * len(args.videos))

        # Handle image directories - modified part
        if args.image_dirs:
            # Make sure image_dirs is a list
            image_dirs = args.image_dirs
            if isinstance(image_dirs, Path):
                image_dirs = [str(image_dirs)]

            for dir_path in image_dirs:
                # Get all subdirectories
                subdirs = get_subdirectories(dir_path)

                if subdirs:
                    # Add all subdirectories as image directories
                    print(f"Found {len(subdirs)} subdirectories in {dir_path}:")
                    for subdir in subdirs:
                        print(f"  - {subdir}")

                    all_inputs.extend(subdirs)
                    input_types.extend([True] * len(subdirs))
                else:
                    # If no subdirectories, process the directory itself
                    print(f"No subdirectories found in {dir_path}, processing it directly.")
                    all_inputs.append(dir_path)
                    input_types.append(True)
        elif not args.videos:  # If no --image-dirs or --videos specified, use default behavior
            # Default behavior: process the 'data' directory if it exists
            data_dir = DEFAULT_DATA_DIR
            if data_dir.exists() and data_dir.is_dir():
                print(f"No input directories specified. Using default data directory: {data_dir}")

                # Get subdirectories in the data directory
                subdirs = get_subdirectories(str(data_dir))

                if subdirs:
                    print(f"Found {len(subdirs)} subdirectories in default data directory:")
                    for subdir in subdirs:
                        print(f"  - {subdir}")

                    all_inputs.extend(subdirs)
                    input_types.extend([True] * len(subdirs))
                else:
                    # Process the data directory itself if no subdirectories
                    print("No subdirectories found in default data directory, processing it directly.")
                    all_inputs.append(str(data_dir))
                    input_types.append(True)

        if all_inputs:
            print("\n=== Processing Videos and Image Sequences ===")
            print(f"Total inputs to process: {len(all_inputs)}")
            for i, path in enumerate(all_inputs):
                print(f"  {i + 1}. {'Image directory' if input_types[i] else 'Video'}: {path}")

            process_videos(
                all_inputs,
                output_dir,
                args.skip_frames,
                args.detect_faces,
                args.detect_pose,
                input_types=input_types,
                image_extension=args.image_extension,
                save_all_frames=args.save_all_frames
            )
        else:
            print("No videos or image sequences to process")

    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()