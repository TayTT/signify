import os
import cv2
import json
import re
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from collections import deque
import math

# TODO: visualise full mesh TODO: not all points present in all frames:
#  1) count missing points frames (how?) and delete them if treshold is not exceeded
#  2)  detect hand disappearing from frame and exclude that from missing points

# TODO fix pose or ignore pose
# TODO fix hands flickering - ADDRESSED WITH ENHANCED TRACKER
# TODO visualize path of the hand
# TODO hands are only moving in 2d, flat to the boy. is there any 3d data?

# MediaPipe solution instances
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Mouth landmark indices from MediaPipe FaceMesh
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]


class EnhancedHandTracker:
    """Enhanced hand tracker with flickering reduction and false positive filtering"""

    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 temporal_smoothing_frames: int = 5,
                 confidence_threshold: float = 0.6,
                 max_hand_distance_threshold: float = 0.3,
                 frame_border_margin: float = 0.1):
        """
        Initialize enhanced hand tracker

        Args:
            min_detection_confidence: Higher threshold for initial detection
            min_tracking_confidence: Threshold for tracking between frames
            temporal_smoothing_frames: Number of frames to consider for smoothing
            confidence_threshold: Minimum confidence to accept detection
            max_hand_distance_threshold: Maximum distance a hand can move between frames
            frame_border_margin: Distance from frame edge to consider "near border" (0.0-0.5)
        """
        self.mp_hands = mp.solutions.hands

        # Initialize MediaPipe with optimized settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Video mode for better tracking
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # Good balance between accuracy and speed
        )

        # Tracking parameters
        self.temporal_smoothing_frames = temporal_smoothing_frames
        self.confidence_threshold = confidence_threshold
        self.max_hand_distance_threshold = max_hand_distance_threshold
        self.frame_border_margin = frame_border_margin

        # Hand tracking history
        self.hand_history = {
            'left_hand': deque(maxlen=temporal_smoothing_frames),
            'right_hand': deque(maxlen=temporal_smoothing_frames)
        }

        # Confidence tracking
        self.confidence_history = {
            'left_hand': deque(maxlen=temporal_smoothing_frames),
            'right_hand': deque(maxlen=temporal_smoothing_frames)
        }

        # Previous frame data for continuity checking
        self.previous_hands = {'left_hand': None, 'right_hand': None}

        # Track hand exit/entry status
        self.hand_exit_status = {
            'left_hand': {'exited_frame': False, 'exit_position': None, 'frames_since_exit': 0},
            'right_hand': {'exited_frame': False, 'exit_position': None, 'frames_since_exit': 0}
        }

        # Frame counter for analysis
        self.frame_count = 0

        # Statistics
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'false_positives_filtered': 0,
            'smoothed_detections': 0,
            'border_exits_detected': 0,
            'invalid_reentries_filtered': 0
        }

    def _is_near_frame_border(self, center: np.ndarray) -> tuple:
        """
        Check if hand center is near frame border

        Args:
            center: Hand center coordinates (normalized 0-1)

        Returns:
            Tuple of (is_near_border, border_side)
            border_side can be 'left', 'right', 'top', 'bottom', or None
        """
        if center is None:
            return False, None

        x, y = center[0], center[1]
        margin = self.frame_border_margin

        # Check each border
        if x <= margin:
            return True, 'left'
        elif x >= (1.0 - margin):
            return True, 'right'
        elif y <= margin:
            return True, 'top'
        elif y >= (1.0 - margin):
            return True, 'bottom'

        return False, None

    def _is_valid_reentry(self, hand_type: str, current_center: np.ndarray) -> bool:
        """
        Check if a hand reentry is valid (near the border where it exited)

        Args:
            hand_type: 'left_hand' or 'right_hand'
            current_center: Current hand center position

        Returns:
            True if reentry is valid, False if it's likely a false positive
        """
        exit_info = self.hand_exit_status[hand_type]

        if not exit_info['exited_frame'] or exit_info['exit_position'] is None:
            return True  # Hand never exited, so any detection is valid

        # Check if current position is near a border
        is_near_border, border_side = self._is_near_frame_border(current_center)

        if is_near_border:
            # Hand is re-entering near a border - this is likely valid
            return True

        # Hand is appearing in middle of frame after exiting
        # This could be a false positive, especially if it happened recently
        frames_since_exit = exit_info['frames_since_exit']

        # Allow reentry in middle if enough frames have passed (hand might have moved off-screen and back)
        max_frames_for_strict_check = 30  # About 1 second at 30fps

        if frames_since_exit > max_frames_for_strict_check:
            return True  # Enough time has passed, allow reentry anywhere

        # Recent exit - be strict about reentry location
        return False

    def _update_exit_status(self, hand_type: str, current_center: np.ndarray, hand_detected: bool):
        """
        Update the exit status tracking for a hand

        Args:
            hand_type: 'left_hand' or 'right_hand'
            current_center: Current hand center (or None if not detected)
            hand_detected: Whether hand was detected in current frame
        """
        exit_info = self.hand_exit_status[hand_type]

        if hand_detected and current_center is not None:
            # Hand is detected
            is_near_border, border_side = self._is_near_frame_border(current_center)

            if exit_info['exited_frame']:
                # Hand was previously marked as exited
                if is_near_border:
                    # Hand is re-entering near border - mark as back in frame
                    exit_info['exited_frame'] = False
                    exit_info['exit_position'] = None
                    exit_info['frames_since_exit'] = 0
                else:
                    # Hand detected in middle after exit - increment counter but keep exit status
                    exit_info['frames_since_exit'] += 1
            else:
                # Hand is in frame, check if it's about to exit
                if is_near_border:
                    # Hand is near border, store position in case it exits next frame
                    exit_info['exit_position'] = current_center.copy()
                # Reset exit status since hand is clearly in frame
                exit_info['frames_since_exit'] = 0
        else:
            # Hand not detected
            if not exit_info['exited_frame'] and exit_info['exit_position'] is not None:
                # Hand was near border and now disappeared - mark as exited
                exit_info['exited_frame'] = True
                exit_info['frames_since_exit'] = 0
                self.stats['border_exits_detected'] += 1
            elif exit_info['exited_frame']:
                # Hand was already marked as exited, increment counter
                exit_info['frames_since_exit'] += 1

    def _get_hand_center(self, landmarks) -> np.ndarray:
        """Get center point of hand (average of all landmarks)"""
        if not landmarks:
            return None

        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        z_coords = [lm.z for lm in landmarks.landmark]

        return np.array([
            np.mean(x_coords),
            np.mean(y_coords),
            np.mean(z_coords)
        ])

    def _calculate_hand_size(self, landmarks) -> float:
        """Calculate hand size based on distance between key points"""
        if not landmarks or len(landmarks.landmark) < 21:
            return 0.0

        # Use distance between wrist and middle finger tip as size measure
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]

        distance = math.sqrt(
            (wrist.x - middle_tip.x) ** 2 +
            (wrist.y - middle_tip.y) ** 2
        )

        return distance

    def _is_valid_hand_size(self, landmarks, min_size: float = 0.05, max_size: float = 0.5) -> bool:
        """Check if detected hand has reasonable size"""
        hand_size = self._calculate_hand_size(landmarks)
        return min_size <= hand_size <= max_size

    def _is_hand_shape_valid(self, landmarks) -> bool:
        """Basic hand shape validation to filter false positives"""
        if not landmarks or len(landmarks.landmark) < 21:
            return False

        # Check if landmarks form a reasonable hand shape
        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]

        # All fingertips should be further from wrist than palm
        palm_center_y = np.mean([landmarks.landmark[i].y for i in [5, 9, 13, 17]])

        # Basic validation: check if fingertips are in reasonable positions relative to palm
        try:
            fingertips_valid = True
            for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
                # Basic sanity check - fingertips shouldn't be exactly at wrist position
                if abs(tip.x - wrist.x) < 0.01 and abs(tip.y - wrist.y) < 0.01:
                    fingertips_valid = False
                    break

            return fingertips_valid
        except:
            return True  # If validation fails, accept the detection
        """Get center point of hand (average of all landmarks)"""
        if not landmarks:
            return None

        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        z_coords = [lm.z for lm in landmarks.landmark]

        return np.array([
            np.mean(x_coords),
            np.mean(y_coords),
            np.mean(z_coords)
        ])

    def _calculate_hand_size(self, landmarks) -> float:
        """Calculate hand size based on distance between key points"""
        if not landmarks or len(landmarks.landmark) < 21:
            return 0.0

        # Use distance between wrist and middle finger tip as size measure
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]

        distance = math.sqrt(
            (wrist.x - middle_tip.x) ** 2 +
            (wrist.y - middle_tip.y) ** 2
        )

        return distance

    def _is_valid_hand_size(self, landmarks, min_size: float = 0.05, max_size: float = 0.5) -> bool:
        """Check if detected hand has reasonable size"""
        hand_size = self._calculate_hand_size(landmarks)
        return min_size <= hand_size <= max_size

    def _is_hand_shape_valid(self, landmarks) -> bool:
        """Basic hand shape validation to filter false positives"""
        if not landmarks or len(landmarks.landmark) < 21:
            return False

        # Check if landmarks form a reasonable hand shape
        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]

        # All fingertips should be further from wrist than palm
        palm_center_y = np.mean([landmarks.landmark[i].y for i in [5, 9, 13, 17]])

        # Basic validation: check if fingertips are in reasonable positions relative to palm
        try:
            fingertips_valid = True
            for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
                # Basic sanity check - fingertips shouldn't be exactly at wrist position
                if abs(tip.x - wrist.x) < 0.01 and abs(tip.y - wrist.y) < 0.01:
                    fingertips_valid = False
                    break

            return fingertips_valid
        except:
            return True  # If validation fails, accept the detection

    def _assign_hand_labels(self, detected_hands: list, handedness_list: list) -> Dict:
        """Improved hand label assignment using spatial consistency"""
        # Always return a dict with both keys, even if some values are None
        result = {'left_hand': None, 'right_hand': None}

        if not detected_hands:
            return result

        if len(detected_hands) == 1:
            # Single hand - use MediaPipe's classification but verify with history
            hand_landmarks, confidence = detected_hands[0]
            hand_label = handedness_list[0].classification[0].label.lower()

            # Check consistency with previous frame if available
            if self.previous_hands.get(f'{hand_label}_hand') is not None:
                current_center = self._get_hand_center(hand_landmarks)
                prev_center = self.previous_hands[f'{hand_label}_hand']['center']

                if prev_center is not None and current_center is not None:
                    distance = np.linalg.norm(current_center - prev_center)

                    # If too far from expected position, might be wrong label
                    if distance > self.max_hand_distance_threshold:
                        # Try the other hand
                        other_label = 'right_hand' if hand_label == 'left' else 'left_hand'
                        if self.previous_hands.get(other_label) is not None:
                            other_prev_center = self.previous_hands[other_label]['center']
                            if other_prev_center is not None:
                                other_distance = np.linalg.norm(current_center - other_prev_center)

                                if other_distance < distance:
                                    hand_label = 'right' if hand_label == 'left' else 'left'

            # Set the detected hand
            result[f'{hand_label}_hand'] = {
                'landmarks': hand_landmarks,
                'confidence': confidence,
                'center': self._get_hand_center(hand_landmarks)
            }
            # The other hand remains None

        elif len(detected_hands) == 2:
            # Two hands - use both MediaPipe classification and spatial reasoning
            hand1_landmarks, confidence1 = detected_hands[0]
            hand2_landmarks, confidence2 = detected_hands[1]

            hand1_label = handedness_list[0].classification[0].label.lower()
            hand2_label = handedness_list[1].classification[0].label.lower()

            center1 = self._get_hand_center(hand1_landmarks)
            center2 = self._get_hand_center(hand2_landmarks)

            # Simple spatial check: left hand should generally be on the left side
            if center1 is not None and center2 is not None:
                if center1[0] > center2[0]:  # center1 is more to the right
                    # Swap if labels don't match spatial positions
                    if hand1_label == 'left' and hand2_label == 'right':
                        hand1_label, hand2_label = hand2_label, hand1_label
                        hand1_landmarks, hand2_landmarks = hand2_landmarks, hand1_landmarks
                        confidence1, confidence2 = confidence2, confidence1
                        center1, center2 = center2, center1

            # Set both hands
            result[f'{hand1_label}_hand'] = {
                'landmarks': hand1_landmarks,
                'confidence': confidence1,
                'center': center1
            }
            result[f'{hand2_label}_hand'] = {
                'landmarks': hand2_landmarks,
                'confidence': confidence2,
                'center': center2
            }

        return result

    def _apply_temporal_smoothing(self, current_hands: Dict) -> Dict:
        """Apply temporal smoothing to reduce flickering"""
        smoothed_hands = {'left_hand': None, 'right_hand': None}

        for hand_type in ['left_hand', 'right_hand']:
            # Safely get the current hand data
            current_hand = current_hands.get(hand_type, None)

            if current_hand is not None:
                # Add to history
                self.hand_history[hand_type].append(current_hand)
                self.confidence_history[hand_type].append(current_hand['confidence'])

                # Calculate average confidence over recent frames
                avg_confidence = np.mean(list(self.confidence_history[hand_type]))

                # Only accept if average confidence is above threshold
                if avg_confidence >= self.confidence_threshold:
                    # Apply position smoothing if we have history
                    if len(self.hand_history[hand_type]) > 1:
                        smoothed_hands[hand_type] = {
                            'landmarks': current_hand['landmarks'],
                            'confidence': avg_confidence,
                            'center': current_hand['center'],
                            'smoothed': True
                        }
                        self.stats['smoothed_detections'] += 1
                    else:
                        smoothed_hands[hand_type] = current_hand
                else:
                    # Filter out low confidence detection
                    self.stats['filtered_detections'] += 1
            else:
                # No detection - gradually reduce confidence history
                if self.confidence_history[hand_type]:
                    self.confidence_history[hand_type].append(0.0)

        return smoothed_hands

    def process_frame(self, rgb_frame: np.ndarray) -> Dict:
        """
        Process a single frame and return enhanced hand tracking results

        Args:
            rgb_frame: Input frame in RGB format (for MediaPipe)

        Returns:
            Dict with hand data suitable for your existing JSON structure
        """
        self.frame_count += 1

        # Get MediaPipe results
        results = self.hands.process(rgb_frame)

        # Initialize frame output
        frame_hands = {'left_hand': None, 'right_hand': None}

        if results.multi_hand_landmarks:
            self.stats['total_detections'] += len(results.multi_hand_landmarks)

            # Filter valid hands
            valid_hands = []
            valid_handedness = []

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                confidence = handedness.classification[0].score
                hand_center = self._get_hand_center(hand_landmarks)

                # Apply validation filters
                if (confidence >= self.confidence_threshold and
                        self._is_valid_hand_size(hand_landmarks) and
                        self._is_hand_shape_valid(hand_landmarks)):

                    valid_hands.append((hand_landmarks, confidence))
                    valid_handedness.append(handedness)
                else:
                    self.stats['false_positives_filtered'] += 1

            # Assign hand labels with improved logic
            if valid_hands:
                current_hands = self._assign_hand_labels(valid_hands, valid_handedness)

                # Apply frame boundary validation for each hand
                for hand_type in ['left_hand', 'right_hand']:
                    hand_data = current_hands.get(hand_type)

                    if hand_data is not None:
                        hand_center = hand_data['center']

                        # Check if this is a valid reentry after hand exited frame
                        if not self._is_valid_reentry(hand_type, hand_center):
                            # Filter out this detection as likely false positive
                            current_hands[hand_type] = None
                            self.stats['invalid_reentries_filtered'] += 1

                        # Update exit status for this hand
                        self._update_exit_status(hand_type, hand_center, hand_data is not None)
                    else:
                        # No hand detected, update exit status
                        self._update_exit_status(hand_type, None, False)

                # Apply temporal smoothing
                frame_hands = self._apply_temporal_smoothing(current_hands)
            else:
                # No valid hands detected, update exit status for both hands
                self._update_exit_status('left_hand', None, False)
                self._update_exit_status('right_hand', None, False)
        else:
            # No hands detected at all, update exit status for both hands
            self._update_exit_status('left_hand', None, False)
            self._update_exit_status('right_hand', None, False)

        # Update previous hands for next frame
        self.previous_hands = {k: v for k, v in frame_hands.items()}

        return frame_hands

    def get_landmarks_for_json(self, hands_data: Dict, frame_shape: tuple) -> Dict:
        """Convert hand data to format suitable for your existing JSON storage"""
        json_data = {'left_hand': [], 'right_hand': []}
        h, w = frame_shape[:2]

        for hand_type, hand_data in hands_data.items():
            if hand_data is not None:
                landmarks = hand_data['landmarks']
                confidence = hand_data['confidence']

                hand_points = []
                for lm in landmarks.landmark:
                    point = {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "px": int(lm.x * w),
                        "py": int(lm.y * h)
                    }
                    hand_points.append(point)

                # Use your existing format but add confidence info
                json_data[hand_type] = {
                    'landmarks': hand_points,
                    'confidence': float(confidence)
                }

        return json_data

    def draw_hands_on_frame(self, frame: np.ndarray, hands_data: Dict):
        """Draw hand landmarks on frame using your existing style"""
        colors = {'left_hand': (0, 255, 0), 'right_hand': (255, 0, 0)}

        for hand_type, hand_data in hands_data.items():
            if hand_data is not None:
                landmarks = hand_data['landmarks']
                confidence = hand_data['confidence']
                is_smoothed = hand_data.get('smoothed', False)

                # Draw landmarks using MediaPipe style
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Add enhanced label
                wrist = landmarks.landmark[0]
                h, w = frame.shape[:2]
                x, y = int(wrist.x * w), int(wrist.y * h)

                label = f"{hand_type.replace('_', ' ').title()}"
                if is_smoothed:
                    label += " (S)"  # Indicate smoothed
                label += f" {confidence:.2f}"

                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[hand_type], 1)

    def close(self):
        """Clean up resources"""
        if self.hands:
            self.hands.close()

    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        total = max(1, self.stats['total_detections'])
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.stats['total_detections'],
            'filtered_detections': self.stats['filtered_detections'],
            'false_positives_filtered': self.stats['false_positives_filtered'],
            'smoothed_detections': self.stats['smoothed_detections'],
            'border_exits_detected': self.stats['border_exits_detected'],
            'invalid_reentries_filtered': self.stats['invalid_reentries_filtered'],
            'filter_rate': (self.stats['filtered_detections'] / total) * 100,
            'false_positive_rate': (self.stats['false_positives_filtered'] / total) * 100,
            'smooth_rate': (self.stats['smoothed_detections'] / total) * 100,
            'border_exit_rate': (self.stats['border_exits_detected'] / max(1, self.frame_count)) * 100,
            'invalid_reentry_rate': (self.stats['invalid_reentries_filtered'] / total) * 100
        }


# Keep your existing enhance_image_for_hand_detection function unchanged
def enhance_image_for_hand_detection(
        image: np.ndarray,
        visualize: bool = False,
        save_comparison: bool = False,
        comparison_save_path: str = None
) -> np.ndarray:
    """
    Enhance the image to improve hand detection.
    [Keep existing implementation unchanged]
    """
    # ... your existing implementation ...
    # Keep original for visualization
    original = image.copy()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Step 3: Convert back to color - FIX: Use proper color conversion
    enhanced_gray_color = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # Step 4: Blend with original for better color preservation
    enhanced_image = cv2.addWeighted(image, 0.7, enhanced_gray_color, 0.3, 0)

    # Save comparison if requested
    if save_comparison and comparison_save_path:
        try:
            # Ensure comparison directory exists
            comparison_dir = Path(comparison_save_path).parent
            comparison_dir.mkdir(parents=True, exist_ok=True)

            # Create side-by-side comparison
            h, w = image.shape[:2]
            comparison_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison_image[:, :w] = original
            comparison_image[:, w:] = enhanced_image

            # Add labels
            cv2.putText(comparison_image, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison_image, "Enhanced", (w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save comparison image
            comparison_img_path = f"{comparison_save_path}_comparison.png"
            cv2.imwrite(comparison_img_path, comparison_image)

            # Create and save histogram comparison
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Original histogram
                ax1.hist(gray.flatten(), 256, [0, 256], color='blue', alpha=0.7)
                ax1.set_title('Original Image Histogram')
                ax1.set_xlabel('Pixel Intensity')
                ax1.set_ylabel('Frequency')

                # Enhanced histogram
                ax2.hist(enhanced_gray.flatten(), 256, [0, 256], color='red', alpha=0.7)
                ax2.set_title('Enhanced Image Histogram')
                ax2.set_xlabel('Pixel Intensity')
                ax2.set_ylabel('Frequency')

                plt.tight_layout()

                # Save histogram
                histogram_path = f"{comparison_save_path}_histogram.png"
                plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
                plt.close()

            except ImportError:
                print("Matplotlib not available. Skipping histogram comparison.")
            except Exception as e:
                print(f"Error creating histogram comparison: {e}")

        except Exception as e:
            print(f"Error saving comparison files: {e}")

    # Visualize all steps if requested (for debugging/development)
    if visualize:
        try:
            import matplotlib.pyplot as plt

            # Create a figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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


# Keep your existing process_image function unchanged
def process_image(
        file_path: str,
        detect_faces: bool = True,
        detect_pose: bool = True,
        use_enhancement: bool = False
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Process a single image and extract all landmarks.
    [Keep existing implementation but can optionally integrate enhanced hand tracking]
    """
    # ... keep your existing implementation unchanged for now ...
    # This function is mainly for single images, so enhanced tracking is less critical

    print(f"=== DEBUG process_image called ===")
    print(f"File path: {file_path}")
    print(f"Use enhancement: {use_enhancement}")
    print(f"Detect faces: {detect_faces}")
    print(f"Detect pose: {detect_pose}")

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None, None

    image = cv2.imread(file_path)
    if image is None:
        print(f"ERROR: Could not read image: {file_path}")
        return None, None

    print(f"Successfully loaded image: {image.shape}")

    image_height, image_width, _ = image.shape

    # Create a copy for annotation (always use original)
    annotated_image = image.copy()

    # Apply enhancement if requested
    if use_enhancement:
        enhanced_image = enhance_image_for_hand_detection(image)
        # Convert enhanced image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        enhanced_rgb = rgb_image  # This is now the enhanced version
    else:
        # Convert original to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        enhanced_rgb = rgb_image

    image_data = {"face": {}, "pose": {}, "hands": {}}

    # Process hands with optimized settings
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
    ) as hands:
        # Use enhanced image for detection if available
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

                    # Draw hand landmarks on original image (not enhanced)
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
            results_face = face_mesh.process(enhanced_rgb)

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

                    # Draw face landmarks on original image
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
            results_pose = pose.process(enhanced_rgb)

            if results_pose.pose_landmarks:
                pose_data = {}
                for landmark in mp_pose.PoseLandmark:
                    lm = results_pose.pose_landmarks.landmark[landmark]
                    pose_data[landmark.name] = {"x": lm.x, "y": lm.y, "z": lm.z}

                image_data["pose"] = pose_data

                # Draw pose landmarks on original image
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

    return image_data, annotated_image


# MODIFIED: Updated process_video function with enhanced hand tracking
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
    NOW WITH ENHANCED HAND TRACKING!
    """
    # ... keep all your existing setup code unchanged until MediaPipe initialization ...
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
    print(f"  - Enhanced hand tracking: ENABLED")

    # MODIFIED: Initialize enhanced hand tracker instead of regular MediaPipe
    enhanced_hand_tracker = EnhancedHandTracker(
        min_detection_confidence=0.7,
        temporal_smoothing_frames=5,
        confidence_threshold=0.6,
        frame_border_margin=0.1  # NEW: 10% of frame width/height considered "near border"
    )

    # Initialize face and pose solutions (keep existing)
    with mp_face_mesh.FaceMesh(
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

        try:
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

                    # Determine if we should save this frame - save every processed frame or just samples
                    if save_all_frames:
                        # Save every processed frame
                        current_frame_path = frames_dir / f"frame_{current_frame_number:04d}.png"
                    else:
                        # Save every 10th processed frame as a sample
                        current_frame_path = frames_dir / f"frame_{current_frame_number:04d}.png" if (
                                processed_frame_count % 10 == 0) else None

                    process_frame_enhanced(
                        frame,
                        current_frame_number,
                        fps,
                        enhanced_hand_tracker,  # Use enhanced tracker
                        face_mesh,
                        pose,
                        extract_face,
                        extract_pose,
                        all_frames_data,
                        annotated_frame_path=current_frame_path,
                        video_writer=video_writer,
                        total_frames=total_frames,
                        skip_frames=skip_frames,
                        save_all_frames=save_all_frames,
                        use_full_mesh=use_full_mesh,
                        use_enhancement=use_enhancement
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
                        # Determine if we should save this frame - save every processed frame or just samples
                        if save_all_frames:
                            # Save every processed frame
                            current_frame_path = frames_dir / f"frame_{frame_count:04d}.png"
                        else:
                            # Save every 10th processed frame as a sample
                            current_frame_path = frames_dir / f"frame_{frame_count:04d}.png" if (
                                    processed_frame_count % 10 == 0) else None

                        process_frame_enhanced(
                            frame,
                            frame_count,
                            fps,
                            enhanced_hand_tracker,  # Use enhanced tracker
                            face_mesh,
                            pose,
                            extract_face,
                            extract_pose,
                            all_frames_data,
                            annotated_frame_path=current_frame_path,
                            video_writer=video_writer,
                            total_frames=total_frames,
                            skip_frames=skip_frames,
                            save_all_frames=save_all_frames,
                            use_full_mesh=use_full_mesh,
                            use_enhancement=use_enhancement
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

        finally:
            # Clean up enhanced hand tracker
            enhanced_hand_tracker.close()

            # Print enhanced tracking statistics
            stats = enhanced_hand_tracker.get_statistics()
            print(f"\n=== Enhanced Hand Tracking Statistics ===")
            print(f"Frames processed: {stats['frames_processed']}")
            print(f"Total hand detections: {stats['total_detections']}")
            print(
                f"False positives filtered: {stats['false_positives_filtered']} ({stats['false_positive_rate']:.1f}%)")
            print(f"Low confidence filtered: {stats['filtered_detections']} ({stats['filter_rate']:.1f}%)")
            print(f"Detections smoothed: {stats['smoothed_detections']} ({stats['smooth_rate']:.1f}%)")

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
            "save_all_frames": save_all_frames,
            "enhanced_hand_tracking": True  # NEW: indicate enhanced tracking was used
        },
        "enhanced_hand_tracking_stats": stats  # NEW: include tracking statistics
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


# NEW: Enhanced process_frame function
def process_frame_enhanced(
        frame, actual_frame_number, fps, enhanced_hand_tracker, face_mesh, pose,
        extract_face, extract_pose, all_frames_data,
        annotated_frame_path=None, video_writer=None,
        total_frames=0, skip_frames=1,
        save_all_frames=False,
        use_full_mesh=False,
        use_enhancement=False
):
    """
    Enhanced frame processing function that uses the new hand tracker
    """
    try:
        # Apply enhancement if requested
        if use_enhancement:
            # Create comparison save path if we're saving frames
            comparison_save_path = None
            if annotated_frame_path:
                # Create comparisons directory alongside frames directory, not inside it
                frames_dir = annotated_frame_path.parent  # This is the "frames" directory
                output_dir = frames_dir.parent  # Go up one level to the main output directory
                comparisons_dir = output_dir / "comparisons"
                comparisons_dir.mkdir(parents=True, exist_ok=True)

                # Set comparison save path (without extension)
                comparison_save_path = str(comparisons_dir / f"frame_{actual_frame_number:04d}")

            # Apply enhancement with comparison saving
            processing_frame = enhance_image_for_hand_detection(
                frame,
                visualize=False,
                save_comparison=(annotated_frame_path is not None),
                comparison_save_path=comparison_save_path
            )
        else:
            processing_frame = frame

        # Convert to RGB for MediaPipe - use the processing frame (enhanced or original)
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        # Make a copy for annotations - always use original frame for annotations
        annotated_frame = frame.copy()

        # Initialize frame data structure - use actual frame number
        frame_data = {
            "frame": actual_frame_number,
            "timestamp": actual_frame_number / fps,
            "hands": {"left_hand": [], "right_hand": []},
            "face": {"all_landmarks": [], "mouth_landmarks": []},
            "pose": {},
            "enhancement_applied": use_enhancement  # Track if enhancement was used
        }

        # Step 1: Process hands with ENHANCED TRACKING
        hands_data = enhanced_hand_tracker.process_frame(rgb_frame)
        json_hands_data = enhanced_hand_tracker.get_landmarks_for_json(hands_data, frame.shape)

        # Update frame data with enhanced hand tracking results
        frame_data["hands"] = json_hands_data

        # Draw enhanced hands on annotated frame
        enhanced_hand_tracker.draw_hands_on_frame(annotated_frame, hands_data)

        # Step 2: Process face landmarks if requested (keep existing implementation)
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
                        # Draw full face mesh on original frame
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    else:
                        # Draw simplified key landmarks (keep your existing implementation)
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
                            cv2.polylines(annotated_frame, [np.array(silhouette_points)], True,
                                          color_map['silhouette'], 1)

                        # Draw eyebrows
                        for idx in KEY_FACE_LANDMARKS['eyebrows']:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(annotated_frame, (x, y), 1, color_map['eyebrows'], -1)

                        # Draw eyes
                        left_eye_points = []
                        left_eye_indices = KEY_FACE_LANDMARKS['eyes'][:16]
                        for idx in left_eye_indices:
                            lm = face_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            left_eye_points.append((x, y))

                        right_eye_points = []
                        right_eye_indices = KEY_FACE_LANDMARKS['eyes'][16:]
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

        # Step 3: Process pose landmarks if requested (keep existing implementation)
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

                # Draw pose landmarks on original frame
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

                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_landmark_spec,
                    connection_drawing_spec=pose_connection_spec
                )

        # Save frame data using actual frame number as key
        all_frames_data[str(actual_frame_number)] = frame_data

        # Add frame info to image - show actual frame number and enhancement status
        progress_percent = (actual_frame_number / total_frames) * 100 if total_frames > 0 else 0
        enhancement_text = " (Enhanced)" if use_enhancement else ""
        tracking_text = " | Enhanced Hand Tracking"
        cv2.putText(
            annotated_frame,
            f"Frame: {actual_frame_number} | {progress_percent:.1f}%{enhancement_text}{tracking_text}",
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


# Keep your existing process_frame function for backwards compatibility if needed
def process_frame(
        frame, actual_frame_number, fps, hands, face_mesh, pose,
        extract_face, extract_pose, all_frames_data,
        annotated_frame_path=None, video_writer=None,
        total_frames=0, skip_frames=1,
        save_all_frames=False,
        use_full_mesh=False,
        use_enhancement=False
):
    """
    LEGACY: Original process_frame function (kept for compatibility)
    NOTE: This is now replaced by process_frame_enhanced for better hand tracking
    """
    # ... keep your existing implementation for backwards compatibility ...
    # This won't be used in the main processing pipeline anymore
    pass


def natural_sort_key(s):
    """
    Sort strings with embedded numbers in natural order.
    For example: frame1.jpg, frame2.jpg, frame10.jpg (instead of frame1.jpg, frame10.jpg, frame2.jpg)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]