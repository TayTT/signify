"""
Enhanced Hand Tracking Module with Flickering Fixes

This module provides improved hand tracking to reduce flickering and false positives.
Key improvements:
1. Temporal smoothing for detection confidence
2. Hand tracking continuity between frames
3. Better false positive filtering
4. Adaptive detection thresholds
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import math


class EnhancedHandTracker:
    """Enhanced hand tracker with flickering reduction and false positive filtering"""

    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 temporal_smoothing_frames: int = 5,
                 confidence_threshold: float = 0.6,
                 max_hand_distance_threshold: float = 0.3):
        """
        Initialize enhanced hand tracker

        Args:
            min_detection_confidence: Higher threshold for initial detection
            min_tracking_confidence: Threshold for tracking between frames
            temporal_smoothing_frames: Number of frames to consider for smoothing
            confidence_threshold: Minimum confidence to accept detection
            max_hand_distance_threshold: Maximum distance a hand can move between frames
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

        # Frame counter for analysis
        self.frame_count = 0

        # Statistics
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'false_positives_filtered': 0,
            'smoothed_detections': 0
        }

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
        # Basic test: fingers should be in expected relative positions

        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]

        # All fingertips should be further from wrist than palm
        palm_center_y = np.mean([landmarks.landmark[i].y for i in [5, 9, 13, 17]])

        fingertips_valid = all(
            tip.y < palm_center_y  # Assuming typical orientation
            for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        )

        return fingertips_valid

    def _assign_hand_labels(self, detected_hands: List[Tuple], handedness_list: List) -> Dict:
        """Improved hand label assignment using spatial consistency"""
        if not detected_hands:
            return {'left_hand': None, 'right_hand': None}

        if len(detected_hands) == 1:
            # Single hand - use MediaPipe's classification but verify with history
            hand_landmarks, confidence = detected_hands[0]
            hand_label = handedness_list[0].classification[0].label.lower()

            # Check consistency with previous frame if available
            if self.previous_hands[f'{hand_label}_hand'] is not None:
                current_center = self._get_hand_center(hand_landmarks)
                prev_center = self.previous_hands[f'{hand_label}_hand']['center']

                if prev_center is not None:
                    distance = np.linalg.norm(current_center - prev_center)

                    # If too far from expected position, might be wrong label
                    if distance > self.max_hand_distance_threshold:
                        # Try the other hand
                        other_label = 'right_hand' if hand_label == 'left' else 'left_hand'
                        if self.previous_hands[other_label] is not None:
                            other_prev_center = self.previous_hands[other_label]['center']
                            other_distance = np.linalg.norm(current_center - other_prev_center)

                            if other_distance < distance:
                                hand_label = 'right' if hand_label == 'left' else 'left'

            return {
                f'{hand_label}_hand': {
                    'landmarks': hand_landmarks,
                    'confidence': confidence,
                    'center': self._get_hand_center(hand_landmarks)
                },
                'right_hand' if hand_label == 'left' else 'left_hand': None
            }

        elif len(detected_hands) == 2:
            # Two hands - use both MediaPipe classification and spatial reasoning
            hand1_landmarks, confidence1 = detected_hands[0]
            hand2_landmarks, confidence2 = detected_hands[1]

            hand1_label = handedness_list[0].classification[0].label.lower()
            hand2_label = handedness_list[1].classification[0].label.lower()

            center1 = self._get_hand_center(hand1_landmarks)
            center2 = self._get_hand_center(hand2_landmarks)

            # Simple spatial check: left hand should generally be on the left side
            if center1[0] > center2[0]:  # center1 is more to the right
                # Swap if labels don't match spatial positions
                if hand1_label == 'left' and hand2_label == 'right':
                    hand1_label, hand2_label = hand2_label, hand1_label
                    hand1_landmarks, hand2_landmarks = hand2_landmarks, hand1_landmarks
                    confidence1, confidence2 = confidence2, confidence1
                    center1, center2 = center2, center1

            return {
                f'{hand1_label}_hand': {
                    'landmarks': hand1_landmarks,
                    'confidence': confidence1,
                    'center': center1
                },
                f'{hand2_label}_hand': {
                    'landmarks': hand2_landmarks,
                    'confidence': confidence2,
                    'center': center2
                }
            }

        return {'left_hand': None, 'right_hand': None}

    def _apply_temporal_smoothing(self, current_hands: Dict) -> Dict:
        """Apply temporal smoothing to reduce flickering"""
        smoothed_hands = {'left_hand': None, 'right_hand': None}

        for hand_type in ['left_hand', 'right_hand']:
            current_hand = current_hands[hand_type]

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
                        # Smooth the center position
                        recent_centers = [h['center'] for h in list(self.hand_history[hand_type])]
                        smoothed_center = np.mean(recent_centers, axis=0)

                        # Use current landmarks but note they're smoothed
                        smoothed_hands[hand_type] = {
                            'landmarks': current_hand['landmarks'],
                            'confidence': avg_confidence,
                            'center': smoothed_center,
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

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Process a single frame and return enhanced hand tracking results

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple of (hand_data_dict, annotated_frame)
        """
        self.frame_count += 1

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get MediaPipe results
        results = self.hands.process(rgb_frame)

        # Initialize frame output
        annotated_frame = frame.copy()
        frame_hands = {'left_hand': None, 'right_hand': None}

        if results.multi_hand_landmarks:
            self.stats['total_detections'] += len(results.multi_hand_landmarks)

            # Filter valid hands
            valid_hands = []
            valid_handedness = []

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                confidence = handedness.classification[0].score

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

                # Apply temporal smoothing
                frame_hands = self._apply_temporal_smoothing(current_hands)

        # Update previous hands for next frame
        self.previous_hands = {k: v for k, v in frame_hands.items()}

        # Draw annotations
        self._draw_hands(annotated_frame, frame_hands)

        # Add statistics overlay
        self._draw_stats_overlay(annotated_frame)

        return frame_hands, annotated_frame

    def _draw_hands(self, frame: np.ndarray, hands_data: Dict):
        """Draw hand landmarks and labels on frame"""
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        colors = {
            'left_hand': (0, 255, 0),  # Green
            'right_hand': (255, 0, 0),  # Blue
        }

        for hand_type, hand_data in hands_data.items():
            if hand_data is not None:
                landmarks = hand_data['landmarks']
                confidence = hand_data['confidence']
                is_smoothed = hand_data.get('smoothed', False)

                # Draw landmarks
                color = colors[hand_type]
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Add label
                wrist = landmarks.landmark[0]
                h, w = frame.shape[:2]
                x, y = int(wrist.x * w), int(wrist.y * h)

                label = f"{hand_type.replace('_', ' ').title()}"
                if is_smoothed:
                    label += " (S)"  # Indicate smoothed
                label += f" {confidence:.2f}"

                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_stats_overlay(self, frame: np.ndarray):
        """Draw tracking statistics overlay"""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        stats_text = [
            f"Frame: {self.frame_count}",
            f"Total detections: {self.stats['total_detections']}",
            f"Filtered: {self.stats['filtered_detections']}",
            f"False positives: {self.stats['false_positives_filtered']}",
            f"Smoothed: {self.stats['smoothed_detections']}"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, y_offset + i * 20),
                        font, 0.4, (0, 255, 255), 1)

    def get_landmarks_for_json(self, hands_data: Dict, frame_shape: tuple) -> Dict:
        """Convert hand data to format suitable for JSON storage"""
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

                json_data[hand_type] = {
                    'landmarks': hand_points,
                    'confidence': float(confidence),
                    'smoothed': hand_data.get('smoothed', False),
                    'calibrated': hand_data.get('calibrated', False)
                }

        return json_data

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
            'filter_rate': (self.stats['filtered_detections'] / total) * 100,
            'false_positive_rate': (self.stats['false_positives_filtered'] / total) * 100,
            'smooth_rate': (self.stats['smoothed_detections'] / total) * 100
        }


# Example usage function
def process_video_with_enhanced_tracking(video_path: str, output_path: str = None):
    """
    Example function showing how to use the enhanced hand tracker
    """
    # Initialize enhanced tracker
    tracker = EnhancedHandTracker(
        min_detection_confidence=0.7,  # Higher threshold
        temporal_smoothing_frames=5,  # Smooth over 5 frames
        confidence_threshold=0.6  # Require sustained confidence
    )

    cap = cv2.VideoCapture(video_path)

    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    all_frame_data = {}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with enhanced tracking
            hands_data, annotated_frame = tracker.process_frame(frame)

            # Store data for JSON export
            json_data = tracker.get_landmarks_for_json(hands_data)
            all_frame_data[str(frame_count)] = {
                'frame': frame_count,
                'hands': json_data
            }

            # Save frame if output specified
            if output_path:
                out.write(annotated_frame)

            # Display frame (optional)
            cv2.imshow('Enhanced Hand Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

        # Print statistics
        stats = tracker.get_statistics()
        print("\nEnhanced Hand Tracking Statistics:")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Filter rate: {stats['filter_rate']:.1f}%")
        print(f"False positive rate: {stats['false_positive_rate']:.1f}%")
        print(f"Smoothing rate: {stats['smooth_rate']:.1f}%")

        tracker.close()

    return all_frame_data, stats