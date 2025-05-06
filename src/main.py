#!/usr/bin/env python3
import argparse
import contextlib
import cv2
import os
import json
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
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
DEFAULT_OUTPUT_DIR = Path("../output_data")
DEFAULT_IMAGE_FILES = ["../data/stockimg.png"]
DEFAULT_VIDEO_FILES = ["../data/samolot.mp4"]
# BG_COLOR = (192, 192, 192)  # Gray background
VISUALIZE_ENHANCEMENTS=False
#
# def parse_arguments() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="MediaPipe-based body tracking for sign language analysis")
#
#     # Input options
#     parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGE_FILES,
#                         help="Paths to input images")
#     parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEO_FILES,
#                         help="Paths to input videos")
#
#     # Processing options
#     parser.add_argument("--skip-image-processing", action="store_true",
#                         help="Skip processing images")
#     parser.add_argument("--skip-video-processing", action="store_true",
#                         help="Skip processing videos")
#     parser.add_argument("--skip-frames", type=int, default=2,
#                         help="Process every nth frame in videos")
#
#     # Feature options
#     parser.add_argument("--detect-faces", dest="detect_faces", action="store_true",
#                         help="Enable face landmark detection (default: enabled)")
#     parser.add_argument("--no-detect-faces", dest="detect_faces", action="store_false",
#                         help="Disable face landmark detection")
#
#     parser.add_argument("--detect-pose", dest="detect_pose", action="store_true",
#                         help="Enable pose landmark detection (default: enabled)")
#     parser.add_argument("--no-detect-pose", dest="detect_pose", action="store_false",
#                         help="Disable pose landmark detection")
#
#     parser.set_defaults(detect_faces=True, detect_pose=True)
#
#     parser.add_argument("--visualize-enhancements", action="store_true",
#                         help="Visualize image enhancement steps")
#
#     # Output options
#     parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
#                         help="Directory to save output files")
#
#     return parser.parse_args()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process images and videos for sign language recognition')

    parser.add_argument('--output-dir', type=str, default='./output',
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
    parser.add_argument('--image-dirs', type=str, nargs='+', default=[],
                        help='List of directories containing image sequences to process as videos')
    parser.add_argument('--image-extension', type=str, default='png',
                        help='File extension for image sequences (jpg, png, etc.)')
    parser.add_argument('--skip-video-processing', action='store_true',
                        help='Skip processing of videos')
    parser.add_argument('--skip-frames', type=int, default=2,
                        help='Process every nth frame in videos')

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
        print(f"Processing image [{idx+1}/{len(image_files)}]: {file_path}")

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
            visualize=VISUALIZE_ENHANCEMENTS
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
#
#
# def process_videos(video_files: List[str], output_dir: Path, skip_frames: int,
#                    extract_face: bool, extract_pose: bool) -> None:
#     """
#     Process multiple video files and extract landmarks.
#
#     Args:
#         video_files: List of video file paths
#         output_dir: Directory to save output files
#         skip_frames: Process every nth frame
#         extract_face: Whether to extract face landmarks
#         extract_pose: Whether to extract pose landmarks
#     """
#     for idx, video_path in enumerate(video_files):
#         video_path = Path(video_path)
#         print(f"Processing video [{idx+1}/{len(video_files)}]: {video_path}")
#
#         if not video_path.exists():
#             print(f"Warning: Video file not found: {video_path}")
#             continue
#
#         # Create a unique output directory for each video
#         video_output_dir = output_dir / f"video_{video_path.stem}"
#
#         process_video(
#             str(video_path),
#             output_dir=str(video_output_dir),
#             skip_frames=skip_frames,
#             extract_face=extract_face,
#             extract_pose=extract_pose
#         )
#
#
# def process_videos(
#         input_paths: List[str],
#         output_dir: Path,
#         skip_frames: int,
#         extract_face: bool,
#         extract_pose: bool,
#         input_types: Optional[List[bool]] = True,
#         image_extension: str = "png"
# ) -> None:
#     """
#     Process multiple video files or image sequence directories and extract landmarks.
#
#     Args:
#         input_paths: List of video file paths or image directory paths
#         output_dir: Directory to save output files
#         skip_frames: Process every nth frame
#         extract_face: Whether to extract face landmarks
#         extract_pose: Whether to extract pose landmarks
#         input_types: List of booleans indicating if each input is an image sequence (True) or video (False).
#                      If None, all inputs are treated as videos.
#         image_extension: File extension to look for when processing image sequences
#     """
#     # Ensure output directory exists
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # If input_types is not provided, assume all inputs are videos
#     if input_types is None:
#         input_types = [False] * len(input_paths)
#
#     #TODO why?
#
#     # Ensure input_types has the same length as input_paths
#     if len(input_types) != len(input_paths):
#         raise ValueError("input_types list must have the same length as input_paths")
#
#     for idx, (input_path, is_image_sequence) in enumerate(zip(input_paths, input_types)):
#         input_path = Path(input_path)
#         input_type = "image directory" if is_image_sequence else "video"
#
#         print(f"Processing {input_type} [{idx+1}/{len(input_paths)}]: {input_path}")
#
#         if not input_path.exists():
#             print(f"Warning: {input_type} not found: {input_path}")
#             continue
#
#         # Create a unique output directory for each input
#         if is_image_sequence:
#             output_subdir = output_dir / f"images_{input_path.name}"
#         else:
#             output_subdir = output_dir / f"video_{input_path.stem}"
#
#         # Process video or image sequence
#         process_video(
#             str(input_path),
#             output_dir=str(output_subdir),
#             skip_frames=skip_frames,
#             extract_face=extract_face,
#             extract_pose=extract_pose,
#             is_image_sequence=is_image_sequence,
#             image_extension=image_extension
#         )
#
def process_videos(
        input_paths: List[str],
        output_dir: Path,
        skip_frames: int,
        extract_face: bool,
        extract_pose: bool,
        input_types: Optional[List[bool]] = True,
        image_extension: str = "png"
) -> None:
    """
    Process multiple video files or image sequence directories and extract landmarks.
    Searches for all image sequence directories if an input path is a directory.

    Args:
        input_paths: List of video file paths or image directory paths
        output_dir: Directory to save output files
        skip_frames: Process every nth frame
        extract_face: Whether to extract face landmarks
        extract_pose: Whether to extract pose landmarks
        input_types: List of booleans indicating if each input is an image sequence (True) or video (False).
                     If None, all inputs are treated as videos.
        image_extension: File extension to look for when processing image sequences
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand input paths to include all image sequence directories
    expanded_input_paths = []
    expanded_input_types = []

    for idx, input_path in enumerate(input_paths):
        path = Path(input_path)

        # Determine input type for this path
        is_image_sequence = True if input_types is True else (
            input_types[idx] if input_types is not None and idx < len(input_types) else False
        )

        # If path is a directory and marked as image sequence, check if we need to search deeper
        if path.is_dir() and is_image_sequence:
            # Check if this directory is an image sequence directory
            has_images = any(path.glob(f"*.{image_extension}"))

            if has_images:
                # This is a valid image sequence directory, add it directly
                expanded_input_paths.append(str(path))
                expanded_input_types.append(True)
            else:
                # Search for subdirectories containing image sequences
                for subdir in path.glob("**/"):
                    if any(subdir.glob(f"*.{image_extension}")):
                        expanded_input_paths.append(str(subdir))
                        expanded_input_types.append(True)
        else:
            # Keep original path (video file or specified image sequence directory)
            expanded_input_paths.append(str(path))
            expanded_input_types.append(is_image_sequence)

    # If no valid paths were found, warn the user
    if not expanded_input_paths:
        print(f"Warning: No valid video files or image sequence directories found in the provided paths.")
        return

    print(f"Found {len(expanded_input_paths)} video files and image sequence directories to process.")

    # Process each expanded path
    for idx, (input_path, is_image_sequence) in enumerate(zip(expanded_input_paths, expanded_input_types)):
        input_path = Path(input_path)
        input_type = "image directory" if is_image_sequence else "video"

        print(f"Processing {input_type} [{idx+1}/{len(expanded_input_paths)}]: {input_path}")

        if not input_path.exists():
            print(f"Warning: {input_type} not found: {input_path}")
            continue

        # Create a unique output directory for each input
        # Use the full path structure to maintain uniqueness and organization
        relative_path = input_path.name
        if is_image_sequence:
            output_subdir = output_dir / f"images_{relative_path}"
        else:
            output_subdir = output_dir / f"video_{input_path.stem}"

        # Process video or image sequence
        process_video(
            str(input_path),
            output_dir=str(output_subdir),
            skip_frames=skip_frames,
            extract_face=extract_face,
            extract_pose=extract_pose,
            is_image_sequence=is_image_sequence,
            image_extension=image_extension
        )

# def main() -> None:
#     """Main function to process images or videos"""
#     # Parse command line arguments
#     args = parse_arguments()
#
#     # Create output directory structure
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     (output_dir / "tmp").mkdir(exist_ok=True)
#
#     landmarks_data = {}
#
#     # Process static images
#     if not args.skip_image_processing and args.images:
#         print("\n=== Processing Images ===")
#         landmarks_data = process_images(
#             args.images,
#             output_dir,
#             args.detect_faces,
#             args.detect_pose,
#             args.visualize_enhancements
#         )
#
#         # Save all landmark data to JSON
#         if landmarks_data:
#             json_output_path = output_dir / "landmarks_data.json"
#             with open(json_output_path, "w") as json_file:
#                 json.dump(landmarks_data, json_file, indent=4)
#
#             print(f"Landmark data saved to {json_output_path}")
#
#     # Process videos
#     if not args.skip_video_processing and args.videos:
#         print("\n=== Processing Videos ===")
#         process_videos(
#             args.videos,
#             output_dir,
#             args.skip_frames,
#             args.detect_faces,
#             args.detect_pose
#         )
#
#     print("\n=== Processing Complete ===")

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
        # Combine all input sources for video processing
        all_inputs = []
        input_types = []

        # Add video files
        if args.videos:
            all_inputs.extend(args.videos)
            input_types.extend([False] * len(args.videos))

        # Add image sequence directories
        if args.image_dirs:
            all_inputs.extend(args.image_dirs)
            input_types.extend([True] * len(args.image_dirs))

        if all_inputs:
            print("\n=== Processing Videos and Image Sequences ===")
            process_videos(
                all_inputs,
                output_dir,
                args.skip_frames,
                args.detect_faces,
                args.detect_pose,
                input_types=input_types,
                image_extension=args.image_extension
            )

    print("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()