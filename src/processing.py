import os
import cv2
import json
import re
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# MediaPipe solution instances
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Mouth landmark indices from MediaPipe FaceMesh
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]


def enhance_image_for_hand_detection(
        image: np.ndarray,
        visualize: bool = False
) -> np.ndarray:
    """
    Enhance the image to improve hand detection.

    Args:
        image: Input image in BGR format
        visualize: If True, displays all processing steps

    Returns:
        Enhanced image
    """
    # Keep original for visualization
    original = image.copy()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Step 3: Convert back to color
    enhanced_gray_color = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    # TODO: the conversion doesn't work for now?

    # Step 4: Blend with original for better color preservation
    enhanced_image = cv2.addWeighted(image, 0.7, enhanced_gray_color, 0.3, 0)


    # TODO: visualise full mesh TODO: not all points present in all frames:
    #  1) count missing points frames (how?) and delete them if treshold is not exceeded
    #  2)  detect hand disappearing from frame and exclude that from missing points

    # Visualize all steps if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt

            # Create a figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Flatten axes for easier indexing
            axes = axes.flatten()

            # Display original image
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Display grayscale image
            axes[1].imshow(gray, cmap='gray')
            axes[1].set_title('Grayscale')
            axes[1].axis('off')

            # Display enhanced grayscale image
            axes[2].imshow(enhanced_gray, cmap='gray')
            axes[2].set_title('Enhanced Grayscale (CLAHE)')
            axes[2].axis('off')

            # Display grayscale back to color
            axes[3].imshow(cv2.cvtColor(enhanced_gray_color, cv2.COLOR_BGR2RGB))
            axes[3].set_title('Grayscale to Color')
            axes[3].axis('off')

            # Display final enhanced image
            axes[4].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
            axes[4].set_title('Final Enhanced Image')
            axes[4].axis('off')

            # Add histograms comparison (original vs enhanced)
            axes[5].hist(gray.flatten(), 256, [0, 256], color='b', alpha=0.5, label='Original')
            axes[5].hist(enhanced_gray.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Enhanced')
            axes[5].set_title('Histogram Comparison')
            axes[5].legend()

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            print(f"Visualization error: {e}")

    return enhanced_image


def process_image(
        file_path: str,
        detect_faces: bool = True,
        detect_pose: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Process a single image and extract all landmarks.

    Args:
        file_path: Path to the image file
        detect_faces: Whether to detect face landmarks
        detect_pose: Whether to detect pose landmarks

    Returns:
        Tuple of (landmarks_data, annotated_image) or (None, None) if processing fails
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None

    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not read image: {file_path}")
        return None, None

    image_height, image_width, _ = image.shape

    # Create a copy for annotation
    annotated_image = image.copy()

    # Enhance image for better hand detection
    enhanced_image = enhance_image_for_hand_detection(image)

    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    image_data = {"face": {}, "pose": {}, "hands": {}}

    # Process hands with optimized settings
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
    ) as hands:
        # Try with both original and enhanced images for better hand detection
        results_hands = hands.process(rgb_image)

        # If no hands detected in original, try with enhanced image
        if not results_hands.multi_hand_landmarks:
            results_hands = hands.process(enhanced_rgb)

        # Extract hand landmarks
        if results_hands.multi_hand_landmarks:
            hands_data = {"left_hand": [], "right_hand": []}

            # If we have handedness information
            if results_hands.multi_handedness:
                for idx, (hand_landmarks, handedness) in enumerate(
                        zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness)
                ):
                    # Get hand type (left or right)
                    hand_label = handedness.classification[0].label.lower()
                    hand_points = []

                    for i, lm in enumerate(hand_landmarks.landmark):
                        # Store normalized coordinates
                        point = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            # Add pixel coordinates for easier use
                            "px": int(lm.x * image_width),
                            "py": int(lm.y * image_height)
                        }
                        hand_points.append(point)

                    hands_data[f"{hand_label}_hand"] = hand_points

                    # Draw hand landmarks with custom settings for better visibility
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Add hand type label
                    wrist_point = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist_point.x * image_width), int(wrist_point.y * image_height)
                    cv2.putText(
                        annotated_image,
                        hand_label.upper(),
                        (wrist_x, wrist_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            image_data["hands"] = hands_data

    # Process face if requested
    if detect_faces:
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,
                min_detection_confidence=0.7
        ) as face_mesh:
            results_face = face_mesh.process(rgb_image)

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    face_data = []
                    mouth_data = []

                    for i, lm in enumerate(face_landmarks.landmark):
                        point = {"x": lm.x, "y": lm.y, "z": lm.z}
                        face_data.append(point)

                        if i in MOUTH_LANDMARKS:
                            mouth_data.append(point)

                    image_data["face"]["all_landmarks"] = face_data
                    image_data["face"]["mouth_landmarks"] = mouth_data

                    # Draw face landmarks
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

    # Process pose if requested
    if detect_pose:
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.7
        ) as pose:
            results_pose = pose.process(rgb_image)

            if results_pose.pose_landmarks:
                pose_data = {}
                for landmark in mp_pose.PoseLandmark:
                    lm = results_pose.pose_landmarks.landmark[landmark]
                    pose_data[landmark.name] = {"x": lm.x, "y": lm.y, "z": lm.z}

                image_data["pose"] = pose_data

                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

    return image_data, annotated_image


# TODO: check points in 3d

def process_video(
        input_path: str,
        output_dir: str = './output_data/video',
        skip_frames: int = 1,
        extract_face: bool = True,
        extract_pose: bool = True,
        is_image_sequence: bool = False,
        image_extension: str = "png",
        save_all_frames: bool = False,
        use_full_mesh: bool = False,
        use_enhancement: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Process video or image sequence for sign language detection including hands, face, and pose landmarks.

    Args:
        input_path: Path to the video file or directory containing image frames
        output_dir: Directory to save output files
        skip_frames: Process every nth frame for performance (default: 1 = process all frames)
        extract_face: Whether to extract face landmarks
        extract_pose: Whether to extract pose landmarks
        is_image_sequence: Whether input is a directory of image frames instead of a video
        image_extension: Image file extension to look for when processing image sequences (jpg, png, etc.)
        save_all_frames: Whether to save all annotated frames to disk
        use_full_mesh: Whether to use full face mesh or simplified key landmarks
        use_enhancement: Whether to apply image enhancement for better hand detection

    Returns:
        Dictionary containing all frame data or None if processing fails
    """
    input_path = Path(input_path)

    # Always set up frames directory
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Frames will be saved to: {frames_dir}")

    if is_image_sequence:
        if not input_path.is_dir():
            print(f"Image directory not found: {input_path}")
            return None

        # Get list of image files with specified extension
        image_files = sorted([f for f in input_path.glob(f"*.{image_extension}")],
                             key=lambda x: natural_sort_key(x.name))

        if not image_files:
            print(f"No {image_extension} images found in directory: {input_path}")
            return None

        total_frames = len(image_files)
        fps = 30  # Default FPS for image sequences

        # Get dimensions from first image
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            print(f"Could not read image: {image_files[0]}")
            return None
        frame_height, frame_width = first_img.shape[:2]

    else:
        # Process as video file
        if not input_path.exists():
            print(f"Video file not found: {input_path}")
            return None

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Could not open video: {input_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output video writer
    output_video_path = output_dir / "annotated_video.mp4"
    print(f"Setting up video writer for: {output_video_path}")

    # Ensure file can be created
    try:
        # Try writing a test file to ensure directory is writable
        test_path = output_dir / "test_write.txt"
        with open(test_path, 'w') as f:
            f.write("Test")
        test_path.unlink()  # Remove test file
        print(f"Directory is writable: {output_dir}")
    except Exception as e:
        print(f"ERROR: Directory is not writable: {output_dir}, error: {str(e)}")
        return None

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Adjust FPS for skipped frames
        output_fps = fps / skip_frames if skip_frames > 1 else fps
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            output_fps,
            (frame_width, frame_height)
        )

        if not video_writer.isOpened():
            print(f"ERROR: Could not initialize video writer for {output_video_path}")
            print(f"Video properties: codec=mp4v, fps={output_fps}, size={frame_width}x{frame_height}")

            # Try with a different codec as fallback
            print("Trying with XVID codec as fallback...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                str(output_dir / "annotated_video.avi"),  # Use .avi extension for XVID
                fourcc,
                output_fps,
                (frame_width, frame_height)
            )

            if not video_writer.isOpened():
                print("ERROR: Could not initialize video writer with fallback codec either")
                return None
            else:
                print("Successfully initialized video writer with fallback codec")
    except Exception as e:
        print(f"ERROR: Exception while initializing video writer: {str(e)}")
        return None

    frame_count = 0  # Actual frame number (0-based from source)
    processed_frame_count = 0  # Number of frames we've actually processed
    all_frames_data = {}

    print(f"Processing settings:")
    print(f"  - Skip frames: {skip_frames}")
    print(f"  - Save all frames: {save_all_frames}")
    print(f"  - Use enhancement: {use_enhancement}")
    print(f"  - Use full mesh: {use_full_mesh}")

    # Initialize all MediaPipe solutions concurrently for efficiency
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1  # Assuming one face for sign language
    ) as face_mesh, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        if is_image_sequence:
            # Process image sequence
            for img_idx, img_path in enumerate(image_files):
                current_frame_number = img_idx  # 0-based frame number

                # Process every nth frame to improve performance
                if skip_frames > 1 and img_idx % skip_frames != 0:
                    continue

                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"Could not read image: {img_path}")
                    continue

                # Apply enhancement if requested
                if use_enhancement:
                    enhanced_frame = enhance_image_for_hand_detection(frame, visualize=False)
                else:
                    enhanced_frame = frame

                # Determine if we should save this frame - save every processed frame or just samples
                if save_all_frames:
                    # Save every processed frame
                    current_frame_path = frames_dir / f"frame_{current_frame_number:04d}.png"
                else:
                    # Save every 10th processed frame as a sample
                    current_frame_path = frames_dir / f"frame_{current_frame_number:04d}.png" if (
                                processed_frame_count % 10 == 0) else None

                process_frame(
                    enhanced_frame, current_frame_number, fps, hands, face_mesh, pose,
                    extract_face, extract_pose, all_frames_data,
                    annotated_frame_path=current_frame_path,
                    video_writer=video_writer,
                    total_frames=total_frames,
                    skip_frames=skip_frames,
                    save_all_frames=save_all_frames,
                    use_full_mesh=use_full_mesh
                )

                processed_frame_count += 1

                # Print progress every 10 processed frames
                if processed_frame_count % 10 == 0:
                    progress_percent = (current_frame_number / total_frames) * 100 if total_frames > 0 else 0
                    print(
                        f"Processed {processed_frame_count} frames (current frame: {current_frame_number}/{total_frames}, {progress_percent:.1f}%)")

        else:
            # Process video file
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame to improve performance
                if frame_count % skip_frames == 0:
                    # Apply enhancement if requested
                    if use_enhancement:
                        enhanced_frame = enhance_image_for_hand_detection(frame, visualize=False)
                    else:
                        enhanced_frame = frame

                    # Determine if we should save this frame - save every processed frame or just samples
                    if save_all_frames:
                        # Save every processed frame
                        current_frame_path = frames_dir / f"frame_{frame_count:04d}.png"
                    else:
                        # Save every 10th processed frame as a sample
                        current_frame_path = frames_dir / f"frame_{frame_count:04d}.png" if (
                                    processed_frame_count % 10 == 0) else None

                    process_frame(
                        enhanced_frame, frame_count, fps, hands, face_mesh, pose,
                        extract_face, extract_pose, all_frames_data,
                        annotated_frame_path=current_frame_path,
                        video_writer=video_writer,
                        total_frames=total_frames,
                        skip_frames=skip_frames,
                        save_all_frames=save_all_frames,
                        use_full_mesh=use_full_mesh
                    )

                    processed_frame_count += 1

                    # Print progress every 10 processed frames
                    if processed_frame_count % 10 == 0:
                        progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        print(
                            f"Processed {processed_frame_count} frames (current frame: {frame_count}/{total_frames}, {progress_percent:.1f}%)")

                frame_count += 1

            # Clean up video capture
            cap.release()

    # Clean up video writer
    video_writer.release()

    # Save metadata
    input_name = input_path.name if not is_image_sequence else input_path.name
    metadata = {
        "input_source": input_name,
        "input_type": "image_sequence" if is_image_sequence else "video",
        "total_frames_in_source": total_frames,
        "total_frames_scanned": frame_count if not is_image_sequence else len(image_files),
        "processed_frames": processed_frame_count,
        "frame_skip": skip_frames,
        "fps": fps,
        "output_fps": fps / skip_frames if skip_frames > 1 else fps,
        "resolution": f"{frame_width}x{frame_height}",
        "components_extracted": {
            "hands": True,
            "face": extract_face,
            "pose": extract_pose
        },
        "processing_options": {
            "enhancement_applied": use_enhancement,
            "full_face_mesh": use_full_mesh,
            "save_all_frames": save_all_frames
        }
    }

    # Save all frame data to JSON
    json_path = output_dir / "video_landmarks.json"
    try:
        with open(json_path, "w") as f:
            json.dump({"metadata": metadata, "frames": all_frames_data}, f, indent=4)
        print(f"Processing complete. Data saved to {json_path}")
        print(f"Total frames in source: {total_frames}")
        print(f"Total frames processed: {processed_frame_count}")
        print(f"Frames saved to disk: {len(list(frames_dir.glob('frame_*.png')))}")

        # Verify output files exist
        expected_video_file = output_video_path
        if not expected_video_file.exists():
            # Check for fallback .avi file
            expected_video_file = output_dir / "annotated_video.avi"

        if expected_video_file.exists():
            print(f"Output video: {expected_video_file} ({expected_video_file.stat().st_size} bytes)")
        else:
            print(f"WARNING: No output video file found")

        return all_frames_data

    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_frame(
        frame, actual_frame_number, fps, hands, face_mesh, pose,
        extract_face, extract_pose, all_frames_data,
        annotated_frame_path=None, video_writer=None,
        total_frames=0, skip_frames=1,
        save_all_frames=False,
        use_full_mesh=False
):
    """
    Helper function to process a single frame

    Args:
        actual_frame_number: The actual frame number from the source (respects skip_frames)
        Other parameters remain the same
    """
    try:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make a copy for annotations
        annotated_frame = frame.copy()

        # Initialize frame data structure - use actual frame number
        frame_data = {
            "frame": actual_frame_number,  # This is now the actual frame number
            "timestamp": actual_frame_number / fps,
            "hands": {"left_hand": [], "right_hand": []},
            "face": {"all_landmarks": [], "mouth_landmarks": []},
            "pose": {}
        }

        # Custom drawing specs for smaller landmarks
        small_landmark_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green
            thickness=1,  # Thinner lines
            circle_radius=1  # Smaller radius
        )

        small_connection_spec = mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Red
            thickness=1,  # Thinner connections
            circle_radius=1  # Smaller radius
        )

        # Custom specs for different landmark types
        hand_landmark_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green
            thickness=1,
            circle_radius=1
        )

        hand_connection_spec = mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Red
            thickness=1,
            circle_radius=1
        )

        face_landmark_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 255),  # Yellow
            thickness=1,
            circle_radius=1
        )

        face_connection_spec = mp_drawing.DrawingSpec(
            color=(0, 0, 255),  # Blue
            thickness=1,
            circle_radius=1
        )

        pose_landmark_spec = mp_drawing.DrawingSpec(
            color=(0, 0, 255),  # Blue
            thickness=1,
            circle_radius=1
        )

        pose_connection_spec = mp_drawing.DrawingSpec(
            color=(255, 255, 0),  # Cyan
            thickness=1,
            circle_radius=1
        )

        # Step 1: Process hands
        results_hands = hands.process(rgb_frame)

        if results_hands.multi_hand_landmarks:
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness)
            ):
                # Get hand type (left or right)
                hand_type = handedness.classification[0].label.lower()
                confidence = handedness.classification[0].score

                # Extract landmarks
                hand_points = []
                h, w, _ = frame.shape
                for i, lm in enumerate(hand_landmarks.landmark):
                    point = {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "px": int(lm.x * w),
                        "py": int(lm.y * h)
                    }
                    hand_points.append(point)

                # Store hand data with confidence score
                hand_data = {
                    "landmarks": hand_points,
                    "confidence": float(confidence)
                }
                frame_data["hands"][f"{hand_type}_hand"] = hand_data

                # Draw hand landmarks with smaller points
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec
                )

                # Add hand label with smaller font
                wrist_point = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist_point.x * w), int(wrist_point.y * h)
                label_text = f"{hand_type.upper()} ({confidence:.2f})"
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (wrist_x, wrist_y - 5),  # Move closer to the landmark
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,  # Smaller font size
                    (0, 255, 0),
                    1  # Thinner text
                )

        # Step 2: Process face landmarks if requested
        if extract_face:
            results_face = face_mesh.process(rgb_frame)

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Extract all face landmarks for JSON data
                    face_data = []
                    mouth_data = []
                    h, w, _ = frame.shape

                    for i, lm in enumerate(face_landmarks.landmark):
                        point = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "px": int(lm.x * w),
                            "py": int(lm.y * h)
                        }
                        face_data.append(point)

                        if i in MOUTH_LANDMARKS:
                            mouth_data.append(point)

                    frame_data["face"]["all_landmarks"] = face_data
                    frame_data["face"]["mouth_landmarks"] = mouth_data

                    # Choose visualization based on use_full_mesh parameter
                    if use_full_mesh:
                        # Draw full face mesh
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    else:
                        # Draw simplified key landmarks
                        KEY_FACE_LANDMARKS = {
                            'silhouette': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                           397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                           172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
                            'eyebrows': [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
                            'eyes': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
                                     362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
                            'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
                            'lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191,
                                     80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178],
                        }

                        color_map = {
                            'silhouette': (200, 200, 200),
                            'eyebrows': (0, 150, 255),
                            'eyes': (255, 0, 0),
                            'nose': (0, 255, 255),
                            'lips': (0, 0, 255)
                        }

                        # Draw face outline (silhouette)
                        silhouette_points = []
                        for idx in KEY_FACE_LANDMARKS['silhouette']:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            silhouette_points.append((x, y))

                        if silhouette_points:
                            # Draw the face contour as a polyline
                            cv2.polylines(annotated_frame, [np.array(silhouette_points)], True,
                                          color_map['silhouette'], 1)

                        # Draw eyebrows
                        for idx in KEY_FACE_LANDMARKS['eyebrows']:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(annotated_frame, (x, y), 1, color_map['eyebrows'], -1)

                        # Draw eyes
                        left_eye_points = []
                        left_eye_indices = KEY_FACE_LANDMARKS['eyes'][:16]  # First 16 are left eye
                        for idx in left_eye_indices:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            left_eye_points.append((x, y))

                        right_eye_points = []
                        right_eye_indices = KEY_FACE_LANDMARKS['eyes'][16:]  # Last 16 are right eye
                        for idx in right_eye_indices:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            right_eye_points.append((x, y))

                        if left_eye_points:
                            cv2.polylines(annotated_frame, [np.array(left_eye_points)], True,
                                          color_map['eyes'], 1)
                        if right_eye_points:
                            cv2.polylines(annotated_frame, [np.array(right_eye_points)], True,
                                          color_map['eyes'], 1)

                        # Draw nose
                        nose_points = []
                        for idx in KEY_FACE_LANDMARKS['nose']:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            nose_points.append((x, y))
                            cv2.circle(annotated_frame, (x, y), 1, color_map['nose'], -1)

                        # Draw lips
                        outer_lip_points = []
                        outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]
                        for idx in outer_lip_indices:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            outer_lip_points.append((x, y))

                        if outer_lip_points:
                            cv2.polylines(annotated_frame, [np.array(outer_lip_points)], True,
                                          color_map['lips'], 1)

                        inner_lip_points = []
                        inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191]
                        for idx in inner_lip_indices:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            inner_lip_points.append((x, y))

                        if inner_lip_points:
                            cv2.polylines(annotated_frame, [np.array(inner_lip_points)], True,
                                          color_map['lips'], 1)

        # Step 3: Process pose landmarks if requested
        if extract_pose:
            results_pose = pose.process(rgb_frame)

            if results_pose.pose_landmarks:
                # Extract pose landmarks
                pose_data = {}
                h, w, _ = frame.shape
                for landmark in mp_pose.PoseLandmark:
                    lm = results_pose.pose_landmarks.landmark[landmark]
                    pose_data[landmark.name] = {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "px": int(lm.x * w),
                        "py": int(lm.y * h),
                        "visibility": float(lm.visibility)
                    }

                frame_data["pose"] = pose_data

                # Draw pose landmarks with smaller points
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_landmark_spec,
                    connection_drawing_spec=pose_connection_spec
                )

        # Save frame data using actual frame number as key
        all_frames_data[str(actual_frame_number)] = frame_data

        # Add frame info to image - show actual frame number
        progress_percent = (actual_frame_number / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(
            annotated_frame,
            f"Frame: {actual_frame_number} | {progress_percent:.1f}%",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        # Save annotated frame image (only if path is provided)
        if annotated_frame_path:
            success = cv2.imwrite(str(annotated_frame_path), annotated_frame)
            if not success:
                print(f"Error: Could not save annotated frame to {annotated_frame_path}")

        # Write frame to video
        if video_writer:
            video_writer.write(annotated_frame)

    except Exception as e:
        print(f"Error processing frame {actual_frame_number}: {e}")
        import traceback
        traceback.print_exc()
def natural_sort_key(s):
    """
    Sort strings with embedded numbers in natural order.
    For example: frame1.jpg, frame2.jpg, frame10.jpg (instead of frame1.jpg, frame10.jpg, frame2.jpg)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
