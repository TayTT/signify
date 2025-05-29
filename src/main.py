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
DEFAULT_OUTPUT_DIR = Path("./output")
DEFAULT_INPUT_DIR = Path("./data")

# TODO: check full mesh option
# TODO: check enhancements

# Also update the argument parser help text to be clearer
def parse_arguments():
    """Parse command line arguments with simplified interface"""
    parser = argparse.ArgumentParser(
        description='Process images and videos for sign language recognition',
        epilog='''
Examples:
  python main.py                                    # Process all data in ./data directory
  python main.py --input-directory ./my_videos     # Process data in custom directory
  python main.py --process-single ./video.mp4      # Process single video file
  python main.py --process-single ./image.jpg      # Process single image file
  python main.py --process-single ./frames_dir     # Process directory of frame images
  python main.py --process-single ./videos_dir     # Process directory of videos
  python main.py --process-single ./mixed_dir      # Process directory with mixed content
  python main.py --full-mesh --enhance             # Use full face mesh and image enhancement
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Core directories
    parser.add_argument('--input-directory', type=str, default=DEFAULT_INPUT_DIR,
                        help='Input directory to search recursively for videos or frame folders (default: ./data)')

    parser.add_argument('--output-directory', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save output files (default: ./output)')

    # Single item processing
    parser.add_argument('--process-single', type=str,
                        help='Process a single item: video file, image file, directory with frames, directory with videos, or mixed directory')

    # Processing options
    parser.add_argument('--full-mesh', action='store_true',
                        help='Display full face mesh instead of simplified key landmarks')

    parser.add_argument('--enhance', action='store_true',
                        help='Apply image enhancement for better hand detection')

    # Advanced options
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Process every nth frame in videos (default: 1 = process all frames)')

    parser.add_argument('--image-extension', type=str, default='png',
                        help='File extension for image sequences (default: png)')

    parser.add_argument('--save-all-frames', action='store_true',
                        help='Save all annotated frames to disk (not just sample frames)')

    return parser.parse_args()


def process_single_item(item_path: Path, output_dir: Path, args) -> None:
    """Process a single video file, image file, image directory, or directory containing videos"""
    print(f"\n=== Single Item Processing ===")
    print(f"Processing: {item_path}")
    print(f"Output directory: {output_dir}")

    if not item_path.exists():
        print(f"ERROR: Path not found: {item_path}")
        return

    # Define file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    if item_path.is_file():
        # Handle single files (video or image)
        file_ext = item_path.suffix.lower()

        if file_ext in video_extensions:
            # Single video file
            print("Detected: Single video file")
            process_single_video_file(item_path, output_dir, args)

        elif file_ext in image_extensions:
            # Single image file
            print("Detected: Single image file")
            process_single_image_file(item_path, output_dir, args)

        else:
            print(f"ERROR: Unsupported file type: {file_ext}")
            print(f"Supported video formats: {', '.join(video_extensions)}")
            print(f"Supported image formats: {', '.join(image_extensions)}")
            return

    elif item_path.is_dir():
        # Handle directories - need to determine what type of directory it is

        # Check for direct image files in the directory (image sequence/frame directory)
        direct_image_files = []
        for ext in image_extensions:
            direct_image_files.extend(list(item_path.glob(f'*{ext}')))

        # Check for direct video files in the directory
        direct_video_files = []
        for ext in video_extensions:
            direct_video_files.extend(list(item_path.glob(f'*{ext}')))

        # Check for subdirectories (might contain videos or images)
        subdirectories = [d for d in item_path.iterdir() if d.is_dir()]

        # Decision logic based on what we found
        if direct_image_files and not direct_video_files:
            # Directory contains only images - treat as image sequence/frames
            print(f"Detected: Image sequence directory with {len(direct_image_files)} images")
            process_single_image_directory(item_path, output_dir, args, direct_image_files)

        elif direct_video_files and not direct_image_files:
            # Directory contains only videos - process all videos
            print(f"Detected: Video directory with {len(direct_video_files)} videos")
            process_multiple_videos_from_directory(direct_video_files, output_dir, args)

        elif direct_image_files and direct_video_files:
            # Directory contains both - ask user or handle based on preference
            print(
                f"Detected: Mixed directory with {len(direct_image_files)} images and {len(direct_video_files)} videos")
            print("Processing videos and treating images as individual files...")

            # Process videos
            if direct_video_files:
                process_multiple_videos_from_directory(direct_video_files, output_dir, args)

            # Process individual images
            for img_file in direct_image_files:
                img_output_dir = output_dir / f"image_{img_file.stem}"
                process_single_image_file(img_file, img_output_dir, args)

        elif subdirectories and not direct_image_files and not direct_video_files:
            # Directory contains only subdirectories - check what's in them
            print(f"Detected: Parent directory with {len(subdirectories)} subdirectories")

            # Analyze subdirectories
            video_subdirs = []
            image_subdirs = []

            for subdir in subdirectories:
                sub_videos = []
                sub_images = []

                for ext in video_extensions:
                    sub_videos.extend(list(subdir.glob(f'*{ext}')))
                for ext in image_extensions:
                    sub_images.extend(list(subdir.glob(f'*{ext}')))

                if sub_videos:
                    video_subdirs.append((subdir, len(sub_videos)))
                if sub_images:
                    image_subdirs.append((subdir, len(sub_images)))

            if video_subdirs:
                print(f"Found {len(video_subdirs)} subdirectories with videos:")
                for subdir, count in video_subdirs:
                    print(f"  - {subdir.name}: {count} videos")

                # Process each video subdirectory
                for subdir, count in video_subdirs:
                    subdir_videos = []
                    for ext in video_extensions:
                        subdir_videos.extend(list(subdir.glob(f'*{ext}')))

                    subdir_output = output_dir / f"videos_{subdir.name}"
                    process_multiple_videos_from_directory(subdir_videos, subdir_output, args)

            if image_subdirs:
                print(f"Found {len(image_subdirs)} subdirectories with images:")
                for subdir, count in image_subdirs:
                    print(f"  - {subdir.name}: {count} images")

                # Process each image subdirectory as an image sequence
                for subdir, count in image_subdirs:
                    subdir_images = []
                    for ext in image_extensions:
                        subdir_images.extend(list(subdir.glob(f'*{ext}')))

                    subdir_output = output_dir / f"images_{subdir.name}"
                    process_single_image_directory(subdir, subdir_output, args, subdir_images)

            if not video_subdirs and not image_subdirs:
                print("ERROR: No video or image files found in any subdirectories")
                return

        else:
            # Empty directory or no recognizable content
            print("ERROR: Directory is empty or contains no video/image files")
            return
    else:
        print(f"ERROR: Path is neither a file nor a directory: {item_path}")
        return


def process_single_image_file(image_path: Path, output_dir: Path, args) -> None:
    """Process a single image file"""
    print(f"Processing single image: {image_path}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use the process_image function for single image processing
        from processing import process_image

        # Process the image
        landmarks_data, annotated_image = process_image(
            str(image_path),
            detect_faces=True,
            detect_pose=True
        )

        if landmarks_data is None:
            print(f"WARNING: Processing failed for {image_path}")
            return

        # Save the annotated image
        annotated_path = output_dir / f"annotated_{image_path.name}"
        if annotated_image is not None:
            cv2.imwrite(str(annotated_path), annotated_image)
            print(f"Saved annotated image: {annotated_path}")

        # Save landmarks data
        json_path = output_dir / f"landmarks_{image_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "metadata": {
                    "input_source": image_path.name,
                    "input_type": "single_image",
                    "processing_options": {
                        "enhancement_applied": args.enhance,
                        "full_face_mesh": args.full_mesh
                    }
                },
                "landmarks": landmarks_data
            }, f, indent=4)

        print(f"Successfully processed single image: {image_path}")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"ERROR: Failed to process single image {image_path}: {str(e)}")


def find_processable_items(input_dir: Path) -> tuple[List[str], List[bool]]:
    """
    Recursively find videos and frame directories in the input directory.

    Args:
        input_dir: Directory to search

    Returns:
        Tuple of (paths, is_image_sequence_flags)
    """
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return [], []

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    found_items = []
    item_types = []  # True for image sequences, False for videos

    print(f"Searching for processable items in: {input_dir}")

    # Walk through directory recursively
    for item in input_dir.rglob('*'):
        if item.is_file() and item.suffix.lower() in video_extensions:
            # Found a video file
            found_items.append(str(item))
            item_types.append(False)
            print(f"Found video: {item}")

        elif item.is_dir():
            # Check if directory contains image files
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(item.glob(f'*{ext}')))

            if image_files:
                # Found a directory with images
                found_items.append(str(item))
                item_types.append(True)
                print(f"Found image directory with {len(image_files)} images: {item}")

    if not found_items:
        print(f"No videos or image directories found in: {input_dir}")
    else:
        print(f"Total items found: {len(found_items)}")

    return found_items, item_types


def process_batch(input_dir: Path, output_dir: Path, args) -> None:
    """Process all items found in the input directory"""
    print(f"\n=== Batch Processing ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Find all processable items
    input_paths, input_types = find_processable_items(input_dir)

    if not input_paths:
        print("No items to process.")
        return

    # Process each item
    for idx, (input_path, is_image_sequence) in enumerate(zip(input_paths, input_types)):
        input_path = Path(input_path)
        input_type = "image directory" if is_image_sequence else "video"

        print(f"\nProcessing {input_type} [{idx + 1}/{len(input_paths)}]: {input_path}")

        # Create output subdirectory
        if is_image_sequence:
            output_subdir = output_dir / f"images_{input_path.name}"
        else:
            output_subdir = output_dir / f"video_{input_path.stem}"

        try:
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Process the item
            result = process_video(
                str(input_path),
                output_dir=str(output_subdir),
                skip_frames=args.skip_frames,
                extract_face=True,  # Always extract face
                extract_pose=True,  # Always extract pose
                is_image_sequence=is_image_sequence,
                image_extension=args.image_extension,
                save_all_frames=args.save_all_frames,
                use_full_mesh=args.full_mesh,
                use_enhancement=args.enhance
            )

            if result is None:
                print(f"WARNING: Processing failed for {input_path}")
            else:
                print(f"Successfully processed {input_path}")

        except Exception as e:
            print(f"ERROR: Failed to process {input_path}: {str(e)}")


def process_single_video_file(video_path: Path, output_dir: Path, args) -> None:
    """Process a single video file"""
    print(f"Detected as video file")
    output_subdir = output_dir / f"video_{video_path.stem}"

    try:
        output_subdir.mkdir(parents=True, exist_ok=True)

        result = process_video(
            str(video_path),
            output_dir=str(output_subdir),
            skip_frames=args.skip_frames,
            extract_face=True,
            extract_pose=True,
            is_image_sequence=False,
            image_extension=args.image_extension,
            save_all_frames=args.save_all_frames,
            use_full_mesh=args.full_mesh,
            use_enhancement=args.enhance
        )

        if result is None:
            print(f"WARNING: Processing failed for {video_path}")
        else:
            print(f"Successfully processed {video_path}")

    except Exception as e:
        print(f"ERROR: Failed to process {video_path}: {str(e)}")


def process_single_image_directory(image_dir: Path, output_dir: Path, args, image_files: list) -> None:
    """Process a single directory containing image sequences"""
    print(f"Detected as image directory with {len(image_files)} images")
    output_subdir = output_dir / f"images_{image_dir.name}"

    try:
        output_subdir.mkdir(parents=True, exist_ok=True)

        result = process_video(
            str(image_dir),
            output_dir=str(output_subdir),
            skip_frames=args.skip_frames,
            extract_face=True,
            extract_pose=True,
            is_image_sequence=True,
            image_extension=args.image_extension,
            save_all_frames=args.save_all_frames,
            use_full_mesh=args.full_mesh,
            use_enhancement=args.enhance
        )

        if result is None:
            print(f"WARNING: Processing failed for {image_dir}")
        else:
            print(f"Successfully processed {image_dir}")

    except Exception as e:
        print(f"ERROR: Failed to process {image_dir}: {str(e)}")


def process_multiple_image_directories(parent_dir: Path, output_dir: Path, args) -> None:
    """Process multiple subdirectories containing image sequences"""
    # Check all common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Find all subdirectories that contain images
    image_dirs = []
    for subdir in parent_dir.iterdir():
        if subdir.is_dir():
            # Check if this subdirectory contains images
            images_in_subdir = []
            for ext in image_extensions:
                images_in_subdir.extend(list(subdir.glob(f'*{ext}')))

            if images_in_subdir:
                image_dirs.append((subdir, len(images_in_subdir)))

    if not image_dirs:
        print(f"ERROR: No subdirectories with images found in: {parent_dir}")
        return

    print(f"Found {len(image_dirs)} subdirectories with images:")
    for img_dir, count in image_dirs:
        print(f"  - {img_dir.name}: {count} images")

    successful_count = 0
    failed_count = 0

    for idx, (img_dir, img_count) in enumerate(image_dirs):
        print(f"\nProcessing image directory [{idx + 1}/{len(image_dirs)}]: {img_dir}")

        # Create output subdirectory for this image sequence
        output_subdir = output_dir / f"images_{img_dir.name}"

        try:
            output_subdir.mkdir(parents=True, exist_ok=True)

            result = process_video(
                str(img_dir),
                output_dir=str(output_subdir),
                skip_frames=args.skip_frames,
                extract_face=True,
                extract_pose=True,
                is_image_sequence=True,
                image_extension=args.image_extension,
                save_all_frames=args.save_all_frames,
                use_full_mesh=args.full_mesh,
                use_enhancement=args.enhance
            )

            if result is None:
                print(f"WARNING: Processing failed for {img_dir}")
                failed_count += 1
            else:
                print(f"Successfully processed {img_dir}")
                successful_count += 1

        except Exception as e:
            print(f"ERROR: Failed to process {img_dir}: {str(e)}")
            failed_count += 1

    print(f"\n=== Processing Summary ===")
    print(f"Total image directories: {len(image_dirs)}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")


def process_multiple_videos_from_directory(video_files: list, output_dir: Path, args) -> None:
    """Process multiple video files found in a directory"""
    print(f"Processing {len(video_files)} video files:")
    for video_file in video_files:
        print(f"  - {video_file}")

    successful_count = 0
    failed_count = 0

    for idx, video_path in enumerate(video_files):
        video_path = Path(video_path)
        print(f"\nProcessing video [{idx + 1}/{len(video_files)}]: {video_path}")

        # Create output subdirectory for this video
        output_subdir = output_dir / f"video_{video_path.stem}"

        try:
            output_subdir.mkdir(parents=True, exist_ok=True)

            result = process_video(
                str(video_path),
                output_dir=str(output_subdir),
                skip_frames=args.skip_frames,
                extract_face=True,
                extract_pose=True,
                is_image_sequence=False,
                image_extension=args.image_extension,
                save_all_frames=args.save_all_frames,
                use_full_mesh=args.full_mesh,
                use_enhancement=args.enhance
            )

            if result is None:
                print(f"WARNING: Processing failed for {video_path}")
                failed_count += 1
            else:
                print(f"Successfully processed {video_path}")
                successful_count += 1

        except Exception as e:
            print(f"ERROR: Failed to process {video_path}: {str(e)}")
            failed_count += 1

    print(f"\n=== Processing Summary ===")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")

    """Process multiple video files found in a directory"""
    print(f"Processing {len(video_files)} video files:")
    for video_file in video_files:
        print(f"  - {video_file}")

    successful_count = 0
    failed_count = 0

    for idx, video_path in enumerate(video_files):
        video_path = Path(video_path)
        print(f"\nProcessing video [{idx + 1}/{len(video_files)}]: {video_path}")

        # Create output subdirectory for this video
        output_subdir = output_dir / f"video_{video_path.stem}"

        try:
            output_subdir.mkdir(parents=True, exist_ok=True)

            result = process_video(
                str(video_path),
                output_dir=str(output_subdir),
                skip_frames=args.skip_frames,
                extract_face=True,
                extract_pose=True,
                is_image_sequence=False,
                image_extension=args.image_extension,
                save_all_frames=args.save_all_frames,
                use_full_mesh=args.full_mesh,
                use_enhancement=args.enhance
            )

            if result is None:
                print(f"WARNING: Processing failed for {video_path}")
                failed_count += 1
            else:
                print(f"Successfully processed {video_path}")
                successful_count += 1

        except Exception as e:
            print(f"ERROR: Failed to process {video_path}: {str(e)}")
            failed_count += 1

    print(f"\n=== Processing Summary ===")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")


def main() -> None:
    """Main function with simplified processing logic"""
    # Parse command line arguments
    args = parse_arguments()

    # Convert paths
    output_dir = Path(args.output_directory)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # Display processing options
    print(f"\n=== Processing Options ===")
    print(f"Full face mesh: {'Yes' if args.full_mesh else 'No (simplified)'}")
    print(f"Image enhancement: {'Yes' if args.enhance else 'No'}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Save all frames: {'Yes' if args.save_all_frames else 'No'}")

    # Process single item or batch
    if args.process_single:
        # Single item processing
        item_path = Path(args.process_single)
        process_single_item(item_path, output_dir, args)
    else:
        # Batch processing
        input_dir = Path(args.input_directory)
        process_batch(input_dir, output_dir, args)

    print(f"\n=== Processing Complete ===")
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()