import argparse
import cv2
import json
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from processing import process_image, process_video, enhance_image_for_hand_detection


# TODO: visualise full mesh TODO: not all points present in all frames:
#  1) count missing points frames (how?) and delete them if treshold is not exceeded -> in processJsons, not processing


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

    parser.add_argument('--track-hands', action='store_true',
                        help='Enable hand path visualization (applied during visualization, not processing)')
    parser.add_argument('--enable-calibration', action='store_true', default=True,
                        help='Enable coordinate calibration to align hands with pose wrists (default: enabled)')

    parser.add_argument('--disable-calibration', action='store_true',
                        help='Disable coordinate calibration (use raw MediaPipe coordinates)')

    parser.add_argument('--disable-mirroring', action='store_true',
                        help='Disable coordinate mirroring during processing')

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
            process_single_video_file_with_calibration(item_path, output_dir, args)

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
    print(f"=== DEBUG: Processing single image ===")
    print(f"Image path: {image_path}")
    print(f"Output dir: {output_dir}")
    print(f"Enhancement enabled: {args.enhance}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        # Save comparisons if enhancement is enabled
        if args.enhance:
            print("Enhancement is enabled, creating comparisons...")
            # Create comparisons directory alongside output directory
            comparisons_dir = output_dir / "comparisons"
            comparisons_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created comparisons directory: {comparisons_dir}")

            # Read image for comparison saving
            image = cv2.imread(str(image_path))
            if image is not None:
                print(f"Successfully read image: {image.shape}")
                # Set comparison save path (without extension)
                comparison_save_path = str(comparisons_dir / f"image_{image_path.stem}")
                print(f"Comparison save path: {comparison_save_path}")

                # Apply enhancement with comparison saving
                try:
                    enhance_image_for_hand_detection(
                        image,
                        visualize=False,
                        save_comparison=True,
                        comparison_save_path=comparison_save_path
                    )
                    print(f"Successfully saved comparison images")
                except Exception as e:
                    print(f"ERROR in enhancement: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"ERROR: Could not read image: {image_path}")

        # Use the updated process_image function with enhancement flag
        print("Calling process_image...")
        try:
            landmarks_data, annotated_image = process_image(
                str(image_path),
                detect_faces=True,
                detect_pose=True,
                use_enhancement=args.enhance
            )
            print(
                f"process_image returned: landmarks_data={'not None' if landmarks_data else 'None'}, annotated_image={'not None' if annotated_image is not None else 'None'}")
        except Exception as e:
            print(f"ERROR in process_image: {e}")
            import traceback
            traceback.print_exc()
            return

        if landmarks_data is None:
            print(f"WARNING: Processing failed for {image_path}")
            return

        # Save the annotated image
        annotated_path = output_dir / f"annotated_{image_path.name}"
        print(f"Saving annotated image to: {annotated_path}")
        if annotated_image is not None:
            success = cv2.imwrite(str(annotated_path), annotated_image)
            if success:
                print(f"Successfully saved annotated image: {annotated_path}")
            else:
                print(f"ERROR: Failed to save annotated image")
        else:
            print("ERROR: annotated_image is None")

        # Save landmarks data
        json_path = output_dir / f"landmarks_{image_path.stem}.json"
        print(f"Saving landmarks to: {json_path}")
        try:
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
            print(f"Successfully saved landmarks JSON")
        except Exception as e:
            print(f"ERROR saving JSON: {e}")

        print(f"=== DEBUG: Successfully processed single image: {image_path} ===")

    except Exception as e:
        print(f"ERROR: Failed to process single image {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()


def process_batch(input_dir: Path, output_dir: Path, args) -> None:
    """Process all items found in the input directory"""
    print(f"\n=== DEBUG: Batch Processing ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Enhancement enabled: {args.enhance}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Save all frames: {args.save_all_frames}")

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Define file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Find all processable items
    found_videos = []
    found_image_dirs = []
    found_single_images = []

    print("Scanning directory structure...")

    # Walk through the directory
    for item in input_dir.rglob('*'):
        print(f"Checking item: {item}")
        if item.is_file():
            file_ext = item.suffix.lower()
            print(f"  File extension: {file_ext}")
            if file_ext in video_extensions:
                found_videos.append(item)
                print(f"  -> Added as video: {item.relative_to(input_dir)}")
            elif file_ext in image_extensions:
                # Check if this image is part of a sequence (in a subdirectory)
                parent_dir = item.parent
                if parent_dir != input_dir:
                    # This is an image in a subdirectory - check if it's part of a sequence
                    sibling_images = [f for f in parent_dir.glob('*') if f.suffix.lower() in image_extensions]
                    if len(sibling_images) > 1:
                        # This is part of an image sequence
                        if parent_dir not in found_image_dirs:
                            found_image_dirs.append(parent_dir)
                            print(
                                f"  -> Added as image directory: {parent_dir.relative_to(input_dir)} ({len(sibling_images)} images)")
                    else:
                        # Single image in subdirectory
                        found_single_images.append(item)
                        print(f"  -> Added as single image: {item.relative_to(input_dir)}")
                else:
                    # Single image in root data directory
                    found_single_images.append(item)
                    print(f"  -> Added as single image: {item.relative_to(input_dir)}")
        elif item.is_dir():
            print(f"  Directory: {item}")

    total_items = len(found_videos) + len(found_image_dirs) + len(found_single_images)
    print(f"\nScan results:")
    print(f"- Videos: {len(found_videos)} -> {[v.name for v in found_videos]}")
    print(f"- Image directories: {len(found_image_dirs)} -> {[d.name for d in found_image_dirs]}")
    print(f"- Single images: {len(found_single_images)} -> {[i.name for i in found_single_images]}")
    print(f"Total items to process: {total_items}")

    if total_items == 0:
        print("No processable items found.")
        return

    processed_count = 0
    failed_count = 0

    # Process videos
    for idx, video_path in enumerate(found_videos):
        print(f"\n=== Processing video [{idx + 1}/{len(found_videos)}]: {video_path.name} ===")

        output_subdir = output_dir / f"video_{video_path.stem}"
        try:
            output_subdir.mkdir(parents=True, exist_ok=True)
            print(f"Created video output dir: {output_subdir}")

            print(f"Calling process_video with enhancement={args.enhance}")
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
                use_enhancement=args.enhance,
                disable_mirroring=args.disable_mirroring,
                args = args
            )

            if result is None:
                print(f"WARNING: process_video returned None for {video_path}")
                failed_count += 1
            else:
                print(f"SUCCESS: process_video completed for {video_path}")
                processed_count += 1

        except Exception as e:
            print(f"ERROR: Exception processing {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1

    # Process image directories
    for idx, img_dir in enumerate(found_image_dirs):
        print(f"\n=== Processing image directory [{idx + 1}/{len(found_image_dirs)}]: {img_dir.name} ===")

        output_subdir = output_dir / f"images_{img_dir.name}"
        try:
            output_subdir.mkdir(parents=True, exist_ok=True)
            print(f"Created image dir output: {output_subdir}")

            print(f"Calling process_video (image sequence) with enhancement={args.enhance}")
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
                use_enhancement=args.enhance,
                disable_mirroring=args.disable_mirroring,
                args=args
            )

            if result is None:
                print(f"WARNING: process_video returned None for {img_dir}")
                failed_count += 1
            else:
                print(f"SUCCESS: process_video completed for {img_dir}")
                processed_count += 1

        except Exception as e:
            print(f"ERROR: Exception processing {img_dir}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1

    # Process single images
    for idx, img_path in enumerate(found_single_images):
        print(f"\n=== Processing single image [{idx + 1}/{len(found_single_images)}]: {img_path.name} ===")

        output_subdir = output_dir / f"image_{img_path.stem}"
        try:
            print(f"Calling process_single_image_file")
            process_single_image_file(img_path, output_subdir, args)
            print(f"SUCCESS: process_single_image_file completed for {img_path}")
            processed_count += 1

        except Exception as e:
            print(f"ERROR: Exception processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1

    print(f"\n=== Processing Summary ===")
    print(f"Total items: {total_items}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")


def process_single_video_file_with_calibration(video_path: Path, output_dir: Path, args) -> None:
    """Process a single video file with calibration"""
    print(f"Processing single video: {video_path}")
    output_subdir = output_dir / f"video_{video_path.stem}"

    try:
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Determine calibration setting
        enable_calibration = args.enable_calibration and not args.disable_calibration

        result = process_video(  # Use new function
            str(video_path),
            output_dir=str(output_subdir),
            skip_frames=args.skip_frames,
            extract_face=True,
            extract_pose=True,
            is_image_sequence=False,
            image_extension=args.image_extension,
            save_all_frames=args.save_all_frames,
            use_full_mesh=args.full_mesh,
            use_enhancement=args.enhance,
            enable_calibration=enable_calibration,
            disable_mirroring=args.disable_mirroring,
            args=args
        )

        if result is None:
            print(f"WARNING: Processing failed for {video_path}")
        else:
            print(f"Successfully processed {video_path}")

    except Exception as e:
        print(f"ERROR: Failed to process {video_path}: {str(e)}")


def process_single_image_directory(image_dir: Path, output_dir: Path, args, image_files: list) -> None:
    """Process a single directory containing image sequences"""
    print(f"Processing image directory: {image_dir} with {len(image_files)} images")
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
            use_enhancement=args.enhance,
            disable_mirroring=args.disable_mirroring,
            args=args
        )

        if result is None:
            print(f"WARNING: Processing failed for {image_dir}")
        else:
            print(f"Successfully processed {image_dir}")

    except Exception as e:
        print(f"ERROR: Failed to process {image_dir}: {str(e)}")


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
                use_enhancement=args.enhance,
                disable_mirroring=args.disable_mirroring,
                args=args
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
    print(f"Mirroring applied: {'No' if args.disable_mirroring else 'Yes'}")

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


def main_with_calibration():
    """Modified main function with calibration support"""
    # Parse arguments with calibration options
    args = parse_arguments()

    # Convert paths
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Display processing options including calibration
    print(f"\n=== Processing Options ===")
    print(f"Full face mesh: {'Yes' if args.full_mesh else 'No (simplified)'}")
    print(f"Image enhancement: {'Yes' if args.enhance else 'No'}")
    print(f"Coordinate calibration: {'Yes' if (args.enable_calibration and not args.disable_calibration) else 'No'}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Save all frames: {'Yes' if args.save_all_frames else 'No'}")
    print(f"Mirroring applied: {'No' if args.disable_mirroring else 'Yes'}")

    # Process based on arguments
    if args.process_single:
        item_path = Path(args.process_single)
        process_single_item(item_path, output_dir, args)
    else:
        input_dir = Path(args.input_directory)
        process_batch(input_dir, output_dir, args)

    print(f"\n=== Processing Complete ===")
    print(f"Results saved to: {output_dir.absolute()}")


# Example calibration validation function:
# def validate_calibration_results(json_path: str):
#     """
#     Validate calibration results in processed JSON data
#     """
#
#
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#
#     frames_data = data.get('frames', {})
#     validation_results = []
#
#     for frame_key, frame_data in frames_data.items():
#         if frame_data.get('calibration_applied', False):
#             metrics = validate_hand_pose_alignment(frame_data)
#             validation_results.append(metrics)
#
#     if validation_results:
#         avg_error = np.mean(
#             [m['average_alignment_error'] for m in validation_results if m['average_alignment_error'] != float('inf')])
#         print(f"Average calibration alignment error: {avg_error:.4f}")
#         print(f"Calibrated frames: {len(validation_results)}")
#     else:
#         print("No calibrated frames found in data")


if __name__ == "__main__":
    main()