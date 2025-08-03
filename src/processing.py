import os
import cv2
import json
import re
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
from collections import deque
import math

# MediaPipe solution instances
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Mouth landmark indices from MediaPipe FaceMesh
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]
DEBUG = False
CORE_POSE_LANDMARKS = [
    'NOSE',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_HIP', 'RIGHT_HIP',
]


# CORE_POSE_LANDMARKS = [
#     'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
#     'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
#     'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
#     'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
#     'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
# ]

class EnhancedHandTracker:
    """Enhanced hand tracker with flickering reduction and false positive filtering"""

    def __init__(self,
                 # min_detection_confidence: float = 0.7,
                 # min_tracking_confidence: float = 0.5,
                 # temporal_smoothing_frames: int = 5,
                 # confidence_threshold: float = 0.6,
                 # max_hand_distance_threshold: float = 0.3,
                 # frame_border_margin: float = 0.1):

                 # min_detection_confidence: float = 0.5,
                 # min_tracking_confidence: float = 0.3,
                 # temporal_smoothing_frames: int = 5,
                 # confidence_threshold: float = 0.4,
                 # max_hand_distance_threshold: float = 0.3,
                 # frame_border_margin: float = 0.1):

                 min_detection_confidence: float = 0.2,
                 min_tracking_confidence: float = 0.2,
                 temporal_smoothing_frames: int = 3,
                 confidence_threshold: float = 0.2,
                 max_hand_distance_threshold: float = 0.4,
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
        self.disappearance_stats = {
            'left_hand': {
                'total_missing_frames': 0,
                'unexpected_disappearances': 0,
                'expected_disappearances': 0,
                'current_missing_streak': 0,
                'max_missing_streak': 0,
                'last_seen_frame': None,
                'disappearance_events': []  # List of disappearance events
            },
            'right_hand': {
                'total_missing_frames': 0,
                'unexpected_disappearances': 0,
                'expected_disappearances': 0,
                'current_missing_streak': 0,
                'max_missing_streak': 0,
                'last_seen_frame': None,
                'disappearance_events': []
            }
        }

        self.last_valid_detections = {
            'left_hand': {'data': None, 'frame': -1},
            'right_hand': {'data': None, 'frame': -1}
        }
        self.max_smoothness_gap = 10  # Only calculate smoothness if gap <= 10 frames

    #-------------- START Quality calculations

    def calculate_hand_quality(self, hand_data: Dict, hand_type: str) -> Dict:
        """Calculate quality without boundary context"""
        if not hand_data or not hand_data.get('landmarks'):
            return {
                'quality': 0.0,
                'confidence': 0.0,
                'completeness': 0.0,
                'stability': 0.0,
                'context': 'no_detection'
            }

        landmarks = hand_data.get('landmarks')

        # Factor 1: Detection confidence
        confidence = hand_data.get('confidence', 0.0)

        # Factor 2: Landmark completeness
        if hasattr(landmarks, 'landmark'):
            # MediaPipe NormalizedLandmarkList object
            completeness = len(landmarks.landmark) / 21.0 if landmarks.landmark else 0.0
        elif isinstance(landmarks, list):
            # Python list (JSON format)
            completeness = len(landmarks) / 21.0 if landmarks else 0.0
        else:
            # Unknown format
            completeness = 0.0

        # Factor 3: Stability (with gap handling)
        stability = self._calculate_stability_with_gaps(hand_data, hand_type)

        # Simple quality calculation using all three factors
        quality_factors = [confidence, completeness, stability]
        weights = [0.5, 0.3, 0.2]  # Confidence most important

        # Calculate weighted quality score
        quality_score = sum(factor * weight for factor, weight in zip(quality_factors, weights))
        quality_score = min(1.0, max(0.0, quality_score))

        # Update last valid detection for future smoothness calculations
        self.last_valid_detections[hand_type] = {
            'data': hand_data,
            'frame': self.frame_count
        }

        return {
            'quality': quality_score,
            'confidence': confidence,
            'completeness': completeness,
            'stability': stability,
            'context': 'detected'
        }

    def _calculate_stability_with_gaps(self, current_hand: Dict, hand_type: str) -> float:
        """Calculate stability handling long periods of missing data"""
        last_valid = self.last_valid_detections[hand_type]

        # No previous valid detection
        if last_valid['data'] is None:
            return 1.0  # Neutral stability for first detection

        # Gap too large - don't calculate stability
        frame_gap = self.frame_count - last_valid['frame']
        if frame_gap > self.max_smoothness_gap:
            return 0.8  # Slightly reduced but not penalized heavily

        # Calculate smoothness between current and last valid detection
        return self._calculate_movement_smoothness(current_hand, last_valid['data'])

    def _calculate_movement_smoothness(self, current_hand: Dict, previous_hand: Dict) -> float:
        """Calculate movement smoothness between two detections"""
        curr_landmarks = current_hand.get('landmarks')
        prev_landmarks = previous_hand.get('landmarks')

        if not curr_landmarks or not prev_landmarks:
            return 0.5  # Neutral stability

        # Handle MediaPipe objects
        def get_wrist_coords(landmarks):
            if hasattr(landmarks, 'landmark'):
                # MediaPipe NormalizedLandmarkList
                if len(landmarks.landmark) > 0:
                    wrist = landmarks.landmark[0]  # Wrist is landmark 0
                    return (wrist.x, wrist.y)
            elif isinstance(landmarks, list) and len(landmarks) > 0:
                # Python list (JSON format)
                wrist = landmarks[0]
                if isinstance(wrist, dict):
                    return (wrist['x'], wrist['y'])
            return None

        curr_wrist_coords = get_wrist_coords(curr_landmarks)
        prev_wrist_coords = get_wrist_coords(prev_landmarks)

        if not curr_wrist_coords or not prev_wrist_coords:
            return 0.5  # Neutral stability

        # Calculate movement distance
        movement_distance = ((curr_wrist_coords[0] - prev_wrist_coords[0]) ** 2 +
                             (curr_wrist_coords[1] - prev_wrist_coords[1]) ** 2) ** 0.5

        # Smooth movement should be moderate (not too fast, not too slow)
        if movement_distance < 0.01:
            return 0.9  # Very stable (minimal movement)
        elif movement_distance < 0.05:
            return 1.0  # Perfect stability (natural movement)
        elif movement_distance < 0.15:
            return 0.8  # Good stability (moderate movement)
        elif movement_distance < 0.3:
            return 0.5  # Reduced stability (fast movement)
        else:
            return 0.2  # Poor stability (very fast/jumpy movement)

    def _get_hand_center(self, landmarks: List[Dict]) -> Tuple[float, float]:
        """Get hand center from landmarks"""
        if not landmarks or len(landmarks) == 0:
            return (0.0, 0.0)

        # Use wrist position as center
        wrist = landmarks[0]
        return (wrist['x'], wrist['y'])

    #------------ END quality calculations
    def _analyze_hand_disappearance(self, hand_type: str, hand_detected: bool, current_center: np.ndarray = None):
        """
        Analyze if hand disappearance is expected (left frame) or unexpected (detection failure)

        Args:
            hand_type: 'left_hand' or 'right_hand'
            hand_detected: Whether hand was detected in current frame
            current_center: Current hand center position (if detected)
        """
        stats = self.disappearance_stats[hand_type]
        exit_info = self.hand_exit_status[hand_type]

        if hand_detected and current_center is not None:
            # Hand is detected
            if stats['current_missing_streak'] > 0:
                # Hand reappeared after being missing
                stats['current_missing_streak'] = 0

            stats['last_seen_frame'] = self.frame_count

        else:
            # Hand not detected
            stats['total_missing_frames'] += 1
            stats['current_missing_streak'] += 1
            stats['max_missing_streak'] = max(stats['max_missing_streak'], stats['current_missing_streak'])

            # Determine if disappearance is expected or unexpected
            if stats['current_missing_streak'] == 1:  # First frame of disappearance
                was_near_border = False

                # Check if hand was near border in recent frames
                if len(self.hand_history[hand_type]) > 0:
                    recent_hand = self.hand_history[hand_type][-1]
                    if recent_hand and 'center' in recent_hand:
                        recent_center = recent_hand['center']
                        is_near_border, border_side = self._is_near_frame_border(recent_center)
                        was_near_border = is_near_border

                # Create disappearance event
                event = {
                    'frame': self.frame_count,
                    'type': 'expected' if was_near_border else 'unexpected',
                    'was_near_border': was_near_border,
                    'border_side': border_side if was_near_border else None,
                    'streak_length': None  # Will be filled when hand reappears
                }

                if was_near_border:
                    stats['expected_disappearances'] += 1
                else:
                    stats['unexpected_disappearances'] += 1

                stats['disappearance_events'].append(event)

            # Update the last event's streak length
            if stats['disappearance_events']:
                stats['disappearance_events'][-1]['streak_length'] = stats['current_missing_streak']

    def process_frame(self, rgb_frame: np.ndarray) -> Dict:
        """Enhanced process_frame with disappearance tracking"""
        self.frame_count += 1

        # Get MediaPipe results
        results = self.hands.process(rgb_frame)

        # Initialize frame output
        frame_hands = {'left_hand': None, 'right_hand': None}

        if results.multi_hand_landmarks:
            self.stats['total_detections'] += len(results.multi_hand_landmarks)

            # Filter valid hands (existing logic)
            valid_hands = []
            valid_handedness = []

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                confidence = handedness.classification[0].score
                hand_center = self._get_hand_center(hand_landmarks)

                # if (confidence >= self.confidence_threshold and
                #         self._is_valid_hand_size(hand_landmarks) and
                #         self._is_hand_shape_valid(hand_landmarks)):
                #
                #     valid_hands.append((hand_landmarks, confidence))
                #     valid_handedness.append(handedness)
                # else:
                #     self.stats['false_positives_filtered'] += 1
                valid_hands.append((hand_landmarks, confidence))
                valid_handedness.append(handedness)

            # Assign hand labels (existing logic)
            if valid_hands:
                current_hands = self._assign_hand_labels(valid_hands, valid_handedness)

                # Apply frame boundary validation (existing logic)
                for hand_type in ['left_hand', 'right_hand']:
                    hand_data = current_hands.get(hand_type)

                    if hand_data is not None:
                        hand_center = hand_data['center']

                        if not self._is_valid_reentry(hand_type, hand_center):
                            current_hands[hand_type] = None
                            self.stats['invalid_reentries_filtered'] += 1

                        self._update_exit_status(hand_type, hand_center, hand_data is not None)
                    else:
                        self._update_exit_status(hand_type, None, False)

                frame_hands = self._apply_temporal_smoothing(current_hands)
            else:
                for hand_type in ['left_hand', 'right_hand']:
                    self._update_exit_status(hand_type, None, False)
        else:
            for hand_type in ['left_hand', 'right_hand']:
                self._update_exit_status(hand_type, None, False)

        # NEW: Analyze disappearances for both hands
        for hand_type in ['left_hand', 'right_hand']:
            hand_data = frame_hands.get(hand_type)
            hand_detected = hand_data is not None
            current_center = hand_data['center'] if hand_data else None

            self._analyze_hand_disappearance(hand_type, hand_detected, current_center)

        # Update previous hands for next frame
        self.previous_hands = {k: v for k, v in frame_hands.items()}

        return frame_hands

    def get_disappearance_statistics(self) -> Dict:
        """Get detailed disappearance statistics"""
        total_frames = max(1, self.frame_count)

        # Calculate frame-by-frame success rates
        left_successful_frames = total_frames - self.disappearance_stats['left_hand']['total_missing_frames']
        right_successful_frames = total_frames - self.disappearance_stats['right_hand']['total_missing_frames']

        return {
            'left_hand': self.disappearance_stats['left_hand'].copy(),
            'right_hand': self.disappearance_stats['right_hand'].copy(),
            'summary': {
                'total_frames_processed': self.frame_count,
                'left_hand_missing_rate': (self.disappearance_stats['left_hand'][
                                               'total_missing_frames'] / total_frames) * 100,
                'right_hand_missing_rate': (self.disappearance_stats['right_hand'][
                                                'total_missing_frames'] / total_frames) * 100,

                # ADD THESE NEW SUCCESS RATES
                'left_hand_success_rate': (left_successful_frames / total_frames) * 100,
                'right_hand_success_rate': (right_successful_frames / total_frames) * 100,

                'total_unexpected_disappearances': (self.disappearance_stats['left_hand']['unexpected_disappearances'] +
                                                    self.disappearance_stats['right_hand'][
                                                        'unexpected_disappearances']),
                'total_expected_disappearances': (self.disappearance_stats['left_hand']['expected_disappearances'] +
                                                  self.disappearance_stats['right_hand']['expected_disappearances'])
            }
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

    # def process_frame_enhanced(
    #         frame, actual_frame_number, fps, enhanced_hand_tracker, face_mesh, pose,
    #         extract_face, extract_pose, all_frames_data,
    #         annotated_frame_path=None, video_writer=None,
    #         total_frames=0, skip_frames=1,
    #         save_all_frames=False,
    #         use_full_mesh=False,
    #         use_enhancement=False
    # ):
    #     """
    #     Enhanced frame processing function that uses the new hand tracker with hand-wrist calibration
    #     """
    #     try:
    #         # Apply enhancement if requested
    #         if use_enhancement:
    #             # Create comparison save path if we're saving frames
    #             comparison_save_path = None
    #             if annotated_frame_path:
    #                 # Create comparisons directory alongside frames directory, not inside it
    #                 frames_dir = annotated_frame_path.parent  # This is the "frames" directory
    #                 output_dir = frames_dir.parent  # Go up one level to the main output directory
    #                 comparisons_dir = output_dir / "comparisons"
    #                 comparisons_dir.mkdir(parents=True, exist_ok=True)
    #
    #                 # Set comparison save path (without extension)
    #                 comparison_save_path = str(comparisons_dir / f"frame_{actual_frame_number:04d}")
    #
    #             # Apply enhancement with comparison saving
    #             processing_frame = enhance_image_for_hand_detection(
    #                 frame,
    #                 visualize=False,
    #                 save_comparison=(annotated_frame_path is not None),
    #                 comparison_save_path=comparison_save_path
    #             )
    #         else:
    #             processing_frame = frame
    #
    #         # Convert to RGB for MediaPipe - use the processing frame (enhanced or original)
    #         rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
    #
    #         # Make a copy for annotations - always use original frame for annotations
    #         annotated_frame = frame.copy()
    #
    #         # Initialize frame data structure - use actual frame number
    #         frame_data = {
    #             "frame": actual_frame_number,
    #             "timestamp": actual_frame_number / fps,
    #             "hands": {"left_hand": [], "right_hand": []},
    #             "face": {"all_landmarks": [], "mouth_landmarks": []},
    #             "pose": {},
    #             "enhancement_applied": use_enhancement  # Track if enhancement was used
    #         }
    #
    #         # Step 1: Process pose landmarks first (needed for hand calibration)
    #         pose_wrists = {"LEFT_WRIST": None, "RIGHT_WRIST": None}
    #
    #         if extract_pose:
    #             results_pose = pose.process(rgb_frame)
    #
    #             if results_pose.pose_landmarks:
    #                 # Extract pose landmarks
    #                 pose_data = {}
    #                 h, w, _ = frame.shape
    #
    #                 for landmark_name in CORE_POSE_LANDMARKS:
    #                     try:
    #                         landmark_enum = getattr(mp_pose.PoseLandmark, landmark_name)
    #                         lm = results_pose.pose_landmarks.landmark[landmark_enum]
    #                         pose_data[landmark_name] = {
    #                             "x": lm.x,
    #                             "y": lm.y,
    #                             "z": lm.z,
    #                             "px": int(lm.x * w),
    #                             "py": int(lm.y * h),
    #                             "visibility": float(lm.visibility)
    #                         }
    #                     except AttributeError:
    #                         continue
    #
    #                 # Store wrist positions for hand calibration
    #                 if "LEFT_WRIST" in pose_data:
    #                     pose_wrists["LEFT_WRIST"] = np.array([
    #                         pose_data["LEFT_WRIST"]["x"],
    #                         pose_data["LEFT_WRIST"]["y"],
    #                         pose_data["LEFT_WRIST"]["z"]
    #                     ])
    #
    #                 if "RIGHT_WRIST" in pose_data:
    #                     pose_wrists["RIGHT_WRIST"] = np.array([
    #                         pose_data["RIGHT_WRIST"]["x"],
    #                         pose_data["RIGHT_WRIST"]["y"],
    #                         pose_data["RIGHT_WRIST"]["z"]
    #                     ])
    #
    #                 frame_data["pose"] = pose_data
    #
    #                 # Draw pose landmarks on original frame
    #                 pose_landmark_spec = mp_drawing.DrawingSpec(
    #                     color=(0, 0, 255),  # Blue
    #                     thickness=1,
    #                     circle_radius=1
    #                 )
    #
    #                 pose_connection_spec = mp_drawing.DrawingSpec(
    #                     color=(255, 255, 0),  # Cyan
    #                     thickness=1,
    #                     circle_radius=1
    #                 )
    #
    #                 mp_drawing.draw_landmarks(
    #                     annotated_frame,
    #                     results_pose.pose_landmarks,
    #                     mp_pose.POSE_CONNECTIONS,
    #                     landmark_drawing_spec=pose_landmark_spec,
    #                     connection_drawing_spec=pose_connection_spec
    #                 )
    #
    #         # Step 2: Process hands with ENHANCED TRACKING and CALIBRATION
    #         hands_data = enhanced_hand_tracker.process_frame(rgb_frame)
    #
    #         # STEP 3: Calibrate hands at MediaPipe object level (BEFORE JSON conversion)
    #         if pose_wrists["LEFT_WRIST"] is not None or pose_wrists["RIGHT_WRIST"] is not None:
    #             # Use the existing calibrate_hands_to_wrists function that works with MediaPipe objects
    #             calibrated_hands_data = calibrate_hands_to_wrists(hands_data, pose_wrists)
    #         else:
    #             calibrated_hands_data = hands_data
    #
    #         # Convert calibrated MediaPipe objects to JSON format
    #         json_hands_data = enhanced_hand_tracker.get_landmarks_for_json(calibrated_hands_data, frame.shape)
    #         frame_data["hands"] = json_hands_data
    #
    #         # Draw calibrated hands on annotated frame
    #         enhanced_hand_tracker.draw_hands_on_frame(annotated_frame, calibrated_hands_data)
    #
    #         # Step 3: Process face landmarks if requested (keep existing implementation)
    #         if extract_face:
    #             results_face = face_mesh.process(rgb_frame)
    #
    #             if results_face.multi_face_landmarks:
    #                 for face_landmarks in results_face.multi_face_landmarks:
    #                     # CALIBRATE face landmarks to pose nose (following hands pattern)
    #                     pose_nose_position = None
    #                     if frame_data["pose"] and "NOSE" in frame_data["pose"]:
    #                         pose_nose_position = np.array([
    #                             frame_data["pose"]["NOSE"]["x"],
    #                             frame_data["pose"]["NOSE"]["y"],
    #                             frame_data["pose"]["NOSE"]["z"]
    #                         ])
    #
    #                     # Calibrate the MediaPipe face landmarks object (same as hands)
    #                     calibrated_face_landmarks = calibrate_face_to_nose(face_landmarks, pose_nose_position)
    #
    #                     # Extract calibrated face landmarks for JSON data
    #                     face_data = []
    #                     mouth_data = []
    #                     h, w, _ = frame.shape
    #
    #                     # Use CALIBRATED landmarks for JSON extraction
    #                     for i, lm in enumerate(calibrated_face_landmarks.landmark):
    #                         point = {
    #                             "x": lm.x,
    #                             "y": lm.y,
    #                             "z": lm.z,
    #                             "px": int(lm.x * w),
    #                             "py": int(lm.y * h)
    #                         }
    #                         face_data.append(point)
    #
    #                         if i in MOUTH_LANDMARKS:
    #                             mouth_data.append(point)
    #
    #                     frame_data["face"]["all_landmarks"] = face_data
    #                     frame_data["face"]["mouth_landmarks"] = mouth_data
    #
    #                     # Mark as calibrated if pose nose was available
    #                     if pose_nose_position is not None:
    #                         frame_data["face"]["calibrated"] = True
    #
    #                     # Draw face landmarks using CALIBRATED MediaPipe object (same as hands)
    #                     if use_full_mesh:
    #                         mp_drawing.draw_landmarks(
    #                             annotated_frame,
    #                             calibrated_face_landmarks,  # ← NOW using calibrated landmarks like hands
    #                             mp_face_mesh.FACEMESH_TESSELATION,
    #                             landmark_drawing_spec=None,
    #                             connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    #                         )
    #                     else:
    #                         # Draw simplified key landmarks (keep your existing implementation)
    #                         KEY_FACE_LANDMARKS = {
    #                             'silhouette': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    #                                            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    #                                            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    #                             'eyebrows': [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    #                             'eyes': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
    #                                      362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381,
    #                                      382],
    #                             'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
    #                             'lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191,
    #                                      80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178],
    #                         }
    #
    #                         color_map = {
    #                             'silhouette': (200, 200, 200),
    #                             'eyebrows': (0, 150, 255),
    #                             'eyes': (255, 0, 0),
    #                             'nose': (0, 255, 255),
    #                             'lips': (0, 0, 255)
    #                         }
    #
    #                         # Draw face outline (silhouette)
    #                         silhouette_points = []
    #                         for idx in KEY_FACE_LANDMARKS['silhouette']:
    #                             lm = face_landmarks.landmark[idx]
    #                             x, y = int(lm.x * w), int(lm.y * h)
    #                             silhouette_points.append((x, y))
    #
    #                         if silhouette_points:
    #                             cv2.polylines(annotated_frame, [np.array(silhouette_points)], True,
    #                                           color_map['silhouette'], 1)
    #
    #                         # Draw eyebrows
    #                         for idx in KEY_FACE_LANDMARKS['eyebrows']:
    #                             lm = face_landmarks.landmark[idx]
    #                             x, y = int(lm.x * w), int(lm.y * h)
    #                             cv2.circle(annotated_frame, (x, y), 1, color_map['eyebrows'], -1)
    #
    #                         # Draw eyes
    #                         left_eye_points = []
    #                         left_eye_indices = KEY_FACE_LANDMARKS['eyes'][:16]
    #                         for idx in left_eye_indices:
    #                             lm = face_landmarks.landmark[idx]
    #                             x, y = int(lm.x * w), int(lm.y * h)
    #                             left_eye_points.append((x, y))
    #
    #                         right_eye_points = []
    #                         right_eye_indices = KEY_FACE_LANDMARKS['eyes'][16:]
    #                         for idx in right_eye_indices:
    #                             lm = face_landmarks.landmark[idx]
    #                             x, y = int(lm.x * w), int(lm.y * h)
    #                             right_eye_points.append((x, y))
    #
    #                         if left_eye_points:
    #                             cv2.polylines(annotated_frame, [np.array(left_eye_points)], True,
    #                                           color_map['eyes'], 1)
    #                         if right_eye_points:
    #                             cv2.polylines(annotated_frame, [np.array(right_eye_points)], True,
    #                                           color_map['eyes'], 1)
    #
    #                         # Draw nose
    #                         nose_points = []
    #                         for idx in KEY_FACE_LANDMARKS['nose']:
    #                             lm = face_landmarks.landmark[idx]
    #                             x, y = int(lm.x * w), int(lm.y * h)
    #                             nose_points.append((x, y))
    #                             cv2.circle(annotated_frame, (x, y), 1, color_map['nose'], -1)
    #
    #                         # Draw lips
    #                         outer_lip_points = []
    #                         outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]
    #                         for idx in outer_lip_indices:
    #                             lm = face_landmarks.landmark[idx]
    #                             x, y = int(lm.x * w), int(lm.y * h)
    #                             outer_lip_points.append((x, y))
    #
    #                         if outer_lip_points:
    #                             cv2.polylines(annotated_frame, [np.array(outer_lip_points)], True,
    #                                           color_map['lips'], 1)
    #
    #                         inner_lip_points = []
    #                         inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191]
    #                         for idx in inner_lip_indices:
    #                             lm = face_landmarks.landmark[idx]
    #                             x, y = int(lm.x * w), int(lm.y * h)
    #                             inner_lip_points.append((x, y))
    #
    #                         if inner_lip_points:
    #                             cv2.polylines(annotated_frame, [np.array(inner_lip_points)], True,
    #                                           color_map['lips'], 1)
    #
    #         # Save frame data using actual frame number as key
    #         all_frames_data[str(actual_frame_number)] = frame_data
    #
    #         # Add frame info to image - show actual frame number and enhancement status
    #         progress_percent = (actual_frame_number / total_frames) * 100 if total_frames > 0 else 0
    #         enhancement_text = " (Enhanced)" if use_enhancement else ""
    #         tracking_text = "  | Enhanced Hand + Face Calibration"
    #         cv2.putText(
    #             annotated_frame,
    #             f"Frame: {actual_frame_number} | {progress_percent:.1f}%{enhancement_text}{tracking_text}",
    #             (10, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0, 255, 255),
    #             1
    #         )
    #
    #         # Save annotated frame image (only if path is provided)
    #         if annotated_frame_path:
    #             success = cv2.imwrite(str(annotated_frame_path), annotated_frame)
    #             if not success:
    #                 print(f"Error: Could not save annotated frame to {annotated_frame_path}")
    #
    #         # Write frame to video
    #         if video_writer:
    #             video_writer.write(annotated_frame)
    #
    #     except Exception as e:
    #         print(f"Error processing frame {actual_frame_number}: {e}")
    #         import traceback
    #         traceback.print_exc()

    def process_frame(self, rgb_frame: np.ndarray) -> Dict:
        """
        Process a single frame and return enhanced hand tracking results
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
                # if (confidence >= self.confidence_threshold and
                #         self._is_valid_hand_size(hand_landmarks) and
                #         self._is_hand_shape_valid(hand_landmarks)):
                #
                #     valid_hands.append((hand_landmarks, confidence))
                #     valid_handedness.append(handedness)
                # else:
                #     self.stats['false_positives_filtered'] += 1
                valid_hands.append((hand_landmarks, confidence))
                valid_handedness.append(handedness)

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

        for hand_type in ['left_hand', 'right_hand']:
            hand_data = frame_hands.get(hand_type)
            hand_detected = hand_data is not None
            current_center = hand_data['center'] if hand_data else None

            self._analyze_hand_disappearance(hand_type, hand_detected, current_center)

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


class FaceTracker:
    """Track face detection and disappearances"""

    def __init__(self, confidence_threshold: float = 0.5, temporal_smoothing_frames: int = 3):
        self.confidence_threshold = confidence_threshold
        self.temporal_smoothing_frames = temporal_smoothing_frames
        self.frame_count = 0

        # Face detection history
        self.face_history = deque(maxlen=temporal_smoothing_frames)
        self.confidence_history = deque(maxlen=temporal_smoothing_frames)

        # Previous frame data
        self.previous_face = None

        # Disappearance tracking
        self.disappearance_stats = {
            'total_missing_frames': 0,
            'unexpected_disappearances': 0,
            'current_missing_streak': 0,
            'max_missing_streak': 0,
            'last_seen_frame': None,
            'disappearance_events': []
        }

        # Detection stats
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'filtered_detections': 0
        }

    def _get_face_center(self, face_landmarks) -> np.ndarray:
        """Get center point of face (nose tip - landmark 1)"""
        if not face_landmarks or len(face_landmarks.landmark) < 2:
            return None

        nose_tip = face_landmarks.landmark[1]  # Nose tip is landmark 1 in MediaPipe
        return np.array([nose_tip.x, nose_tip.y, nose_tip.z])

    def _analyze_face_disappearance(self, face_detected: bool, current_center: np.ndarray = None):
        """
        Analyze if face disappearance is expected or unexpected

        Args:
            face_detected: Whether face was detected in current frame
            current_center: Current face center position (if detected)
        """
        stats = self.disappearance_stats

        if face_detected and current_center is not None:
            # Face is detected
            if stats['current_missing_streak'] > 0:
                # Face reappeared after being missing
                stats['current_missing_streak'] = 0

            stats['last_seen_frame'] = self.frame_count

        else:
            # Face not detected
            stats['total_missing_frames'] += 1
            stats['current_missing_streak'] += 1
            stats['max_missing_streak'] = max(stats['max_missing_streak'], stats['current_missing_streak'])

            # For faces, most disappearances are unexpected since faces shouldn't leave frame easily
            if stats['current_missing_streak'] == 1:  # First frame of disappearance
                # Create disappearance event
                event = {
                    'frame': self.frame_count,
                    'type': 'unexpected',  # Most face disappearances are unexpected
                    'streak_length': None  # Will be filled when face reappears
                }

                stats['unexpected_disappearances'] += 1
                stats['disappearance_events'].append(event)

            # Update the last event's streak length
            if stats['disappearance_events']:
                stats['disappearance_events'][-1]['streak_length'] = stats['current_missing_streak']

    def process_face_detection(self, face_landmarks, confidence: float = 1.0) -> Dict:
        """
        Process face detection results for a frame

        Args:
            face_landmarks: MediaPipe face landmarks (or None if not detected)
            confidence: Detection confidence score

        Returns:
            Dictionary with face detection info (or None if face rejected)
        """
        self.frame_count += 1

        face_detected = face_landmarks is not None
        current_center = None
        result = None

        if face_detected:
            self.detection_stats['total_detections'] += 1
            current_center = self._get_face_center(face_landmarks)

            # Add to history
            face_data = {
                'landmarks': face_landmarks,
                'confidence': confidence,
                'center': current_center
            }

            self.face_history.append(face_data)
            self.confidence_history.append(confidence)

            # Check if detection should be accepted
            avg_confidence = np.mean(list(self.confidence_history))

            if avg_confidence >= self.confidence_threshold:
                self.detection_stats['successful_detections'] += 1
                result = face_data
                face_detected = True  # Keep as detected
            else:
                self.detection_stats['filtered_detections'] += 1
                face_detected = False  # Mark as not detected due to low confidence
                result = None
        else:
            # No face detected
            if self.confidence_history:
                self.confidence_history.append(0.0)

        # CRITICAL: Analyze disappearance regardless of confidence filtering
        self._analyze_face_disappearance(face_detected, current_center)

        # Update previous face
        self.previous_face = result

        return result

    def get_disappearance_statistics(self) -> Dict:
        """Get detailed face disappearance statistics matching hand statistics format"""
        total_frames = max(1, self.frame_count)
        successful_frames = total_frames - self.disappearance_stats['total_missing_frames']

        return {
            'face': {
                'total_missing_frames': self.disappearance_stats['total_missing_frames'],
                'unexpected_disappearances': self.disappearance_stats['unexpected_disappearances'],
                'expected_disappearances': 0,  # Face disappearances are typically unexpected
                'current_missing_streak': self.disappearance_stats['current_missing_streak'],
                'max_missing_streak': self.disappearance_stats['max_missing_streak'],
                'last_seen_frame': self.disappearance_stats['last_seen_frame'],
                'disappearance_events': self.disappearance_stats['disappearance_events']
            },
            'summary': {
                'total_frames_processed': self.frame_count,
                'face_missing_rate': (self.disappearance_stats['total_missing_frames'] / total_frames) * 100,
                'face_success_rate': (successful_frames / total_frames) * 100,
                'total_unexpected_disappearances': self.disappearance_stats['unexpected_disappearances'],
                'total_expected_disappearances': 0
            },
            'detection_stats': self.detection_stats.copy()
        }


def apply_mirroring_to_frame_data(frame_data: Dict, pose_wrists: Dict, frame_shape: tuple) -> Dict:
    """
    Apply mirroring to already-calibrated frame data - SIMPLE VERSION that works

    Just mirror coordinates like face does - no complex hand swapping needed!
    """
    h, w = frame_shape[:2]
    mirrored_frame_data = frame_data.copy()

    # Mirror pose landmarks
    if 'pose' in frame_data:
        mirrored_pose = {}
        for landmark_name, lm_data in frame_data['pose'].items():
            mirrored_lm = lm_data.copy()
            mirrored_lm['x'] = 1.0 - lm_data['x']
            mirrored_lm['px'] = int(mirrored_lm['x'] * w)
            mirrored_pose[landmark_name] = mirrored_lm
        mirrored_frame_data['pose'] = mirrored_pose

    # Mirror face landmarks (this was working correctly)
    if 'face' in frame_data:
        mirrored_face = frame_data['face'].copy()

        if 'all_landmarks' in frame_data['face']:
            mirrored_face_landmarks = []
            for lm in frame_data['face']['all_landmarks']:
                mirrored_lm = lm.copy()
                mirrored_lm['x'] = 1.0 - lm['x']
                mirrored_lm['px'] = int(mirrored_lm['x'] * w)
                mirrored_face_landmarks.append(mirrored_lm)
            mirrored_face['all_landmarks'] = mirrored_face_landmarks

        if 'mouth_landmarks' in frame_data['face']:
            mirrored_mouth_landmarks = []
            for lm in frame_data['face']['mouth_landmarks']:
                mirrored_lm = lm.copy()
                mirrored_lm['x'] = 1.0 - lm['x']
                mirrored_lm['px'] = int(mirrored_lm['x'] * w)
                mirrored_mouth_landmarks.append(mirrored_lm)
            mirrored_face['mouth_landmarks'] = mirrored_mouth_landmarks

        mirrored_frame_data['face'] = mirrored_face

    # Mirror hands EXACTLY like face - simple coordinate mirroring, no swapping
    if 'hands' in frame_data:
        mirrored_hands = frame_data['hands'].copy()

        for hand_type in ['left_hand', 'right_hand']:
            hand_data = frame_data['hands'].get(hand_type, [])

            if hand_data:
                if isinstance(hand_data, dict) and 'landmarks' in hand_data:
                    # Handle new format with confidence
                    mirrored_landmarks = []
                    for lm in hand_data['landmarks']:
                        mirrored_lm = lm.copy()
                        mirrored_lm['x'] = 1.0 - lm['x']
                        mirrored_lm['px'] = int(mirrored_lm['x'] * w)
                        mirrored_landmarks.append(mirrored_lm)

                    mirrored_hands[hand_type] = hand_data.copy()
                    mirrored_hands[hand_type]['landmarks'] = mirrored_landmarks

                elif isinstance(hand_data, list):
                    # Handle old format - direct list of landmarks
                    mirrored_landmarks = []
                    for lm in hand_data:
                        mirrored_lm = lm.copy()
                        mirrored_lm['x'] = 1.0 - lm['x']
                        mirrored_lm['px'] = int(mirrored_lm['x'] * w)
                        mirrored_landmarks.append(mirrored_lm)

                    mirrored_hands[hand_type] = mirrored_landmarks

        mirrored_frame_data['hands'] = mirrored_hands

    # Keep missing_data as-is (no swapping needed with simple approach)
    if 'missing_data' in frame_data:
        mirrored_frame_data['missing_data'] = frame_data['missing_data'].copy()

    # Mark as mirrored
    mirrored_frame_data['mirrored'] = True

    return mirrored_frame_data


def mirror_single_hand_data(hand_data, frame_width: int):
    """
    Mirror a single hand's data (coordinates only)

    Args:
        hand_data: Hand landmarks data (list or dict format)
        frame_width: Frame width for pixel coordinate conversion

    Returns:
        Mirrored hand data in the same format
    """
    if not hand_data:
        return []

    if isinstance(hand_data, dict) and 'landmarks' in hand_data:
        # Handle new format with confidence
        mirrored_landmarks = []
        for lm in hand_data['landmarks']:
            mirrored_lm = lm.copy()
            mirrored_lm['x'] = 1.0 - lm['x']
            mirrored_lm['px'] = int(mirrored_lm['x'] * frame_width)
            mirrored_landmarks.append(mirrored_lm)

        mirrored_hand = hand_data.copy()
        mirrored_hand['landmarks'] = mirrored_landmarks
        return mirrored_hand

    elif isinstance(hand_data, list):
        # Handle old format - direct list of landmarks
        mirrored_landmarks = []
        for lm in hand_data:
            mirrored_lm = lm.copy()
            mirrored_lm['x'] = 1.0 - lm['x']
            mirrored_lm['px'] = int(mirrored_lm['x'] * frame_width)
            mirrored_landmarks.append(mirrored_lm)
        return mirrored_landmarks

    return []


def calibrate_hands_to_wrists(hands_data: Dict, pose_wrists: Dict) -> Dict:
    """
    Calibrate hand positions to align with pose model wrist positions

    Args:
        hands_data: Hand data from enhanced hand tracker
        pose_wrists: Dictionary with LEFT_WRIST and RIGHT_WRIST positions from pose model

    Returns:
        Calibrated hands data with adjusted positions
    """
    calibrated_hands = {'left_hand': None, 'right_hand': None}

    # Mapping between hand types and pose wrist names
    hand_to_wrist_mapping = {
        'left_hand': 'RIGHT_WRIST',
        'right_hand': 'LEFT_WRIST'
    }

    for hand_type in ['left_hand', 'right_hand']:
        hand_data = hands_data.get(hand_type)
        wrist_name = hand_to_wrist_mapping[hand_type]
        pose_wrist_position = pose_wrists.get(wrist_name)

        if hand_data is not None and pose_wrist_position is not None:
            # Get the current hand wrist position (landmark 0)
            hand_landmarks = hand_data['landmarks']
            hand_wrist = hand_landmarks.landmark[0]  # Wrist is landmark 0 in MediaPipe hands

            current_hand_wrist = np.array([hand_wrist.x, hand_wrist.y, hand_wrist.z])

            # Calculate the translation offset needed
            translation_offset = pose_wrist_position - current_hand_wrist

            # Create new landmarks structure with calibrated positions
            calibrated_landmarks = type(hand_landmarks)()
            calibrated_landmarks.CopyFrom(hand_landmarks)

            # Apply translation to all hand landmarks
            for i, landmark in enumerate(calibrated_landmarks.landmark):
                landmark.x += translation_offset[0]
                landmark.y += translation_offset[1]
                landmark.z += translation_offset[2]

            # Create calibrated hand data
            calibrated_hands[hand_type] = {
                'landmarks': calibrated_landmarks,
                'confidence': hand_data['confidence'],
                'center': hand_data['center'] + translation_offset,  # Update center too
                'smoothed': hand_data.get('smoothed', False),
                'calibrated': True  # Mark as calibrated
            }
        elif hand_data is not None:
            # Keep original hand data if no pose wrist available
            calibrated_hands[hand_type] = hand_data

    return calibrated_hands


def calibrate_face_to_nose(face_landmarks, pose_nose_position):
    """
    Calibrate face landmarks to align with pose nose position
    Following the same pattern as calibrate_hands_to_wrists
    """
    if face_landmarks is None or pose_nose_position is None:
        return face_landmarks

    # Get face nose position (MediaPipe face landmark 1 is nose tip)
    face_nose = face_landmarks.landmark[1]  # Nose tip
    face_nose_position = np.array([face_nose.x, face_nose.y, face_nose.z])

    # Calculate translation offset (same as hands)
    translation_offset = pose_nose_position - face_nose_position

    # Create calibrated landmarks structure (same as hands approach)
    calibrated_landmarks = type(face_landmarks)()
    calibrated_landmarks.CopyFrom(face_landmarks)

    # Apply translation to all face landmarks (same as hands approach)
    for landmark in calibrated_landmarks.landmark:
        landmark.x += translation_offset[0]
        landmark.y += translation_offset[1]
        landmark.z += translation_offset[2]

    return calibrated_landmarks


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
    # if DEBUG:
    #     print(f"=== DEBUG process_image called ===")
    #     print(f"File path: {file_path}")
    #     print(f"Use enhancement: {use_enhancement}")
    #     print(f"Detect faces: {detect_faces}")
    #     print(f"Detect pose: {detect_pose}")

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
                    # Apply calibration if pose data is available
                    calibrated_face_landmarks = face_landmarks
                    if detect_pose and "pose" in image_data and "NOSE" in image_data["pose"]:
                        pose_nose_position = np.array([
                            image_data["pose"]["NOSE"]["x"],
                            image_data["pose"]["NOSE"]["y"],
                            image_data["pose"]["NOSE"]["z"]
                        ])
                        calibrated_face_landmarks = calibrate_face_to_nose(face_landmarks, pose_nose_position)
                        image_data["face"]["calibrated"] = True

                    # Extract JSON data from calibrated landmarks
                    face_data = []
                    mouth_data = []
                    for i, lm in enumerate(calibrated_face_landmarks.landmark):
                        point = {"x": lm.x, "y": lm.y, "z": lm.z}
                        face_data.append(point)

                        if i in MOUTH_LANDMARKS:
                            mouth_data.append(point)

                    image_data["face"]["all_landmarks"] = face_data
                    image_data["face"]["mouth_landmarks"] = mouth_data

                    # Draw calibrated face landmarks
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        calibrated_face_landmarks,  # ← Use calibrated landmarks
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

                for landmark_name in CORE_POSE_LANDMARKS:
                    try:
                        landmark_enum = getattr(mp_pose.PoseLandmark, landmark_name)
                        lm = results_pose.pose_landmarks.landmark[landmark_enum]
                        pose_data[landmark_name] = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": float(lm.visibility)
                        }
                    except AttributeError:
                        continue

                image_data["pose"] = pose_data

                # Draw pose landmarks on original image
                if results_pose.pose_landmarks:
                    # Create a subset of landmarks to draw
                    for landmark_name in CORE_POSE_LANDMARKS:
                        try:
                            landmark_enum = getattr(mp_pose.PoseLandmark, landmark_name)
                            lm = results_pose.pose_landmarks.landmark[landmark_enum]
                            x, y = int(lm.x * image_width), int(lm.y * image_height)
                            cv2.circle(annotated_image, (x, y), 2, (0, 0, 255), -1)  # Blue dots
                        except AttributeError:
                            continue
    return image_data, annotated_image


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
        use_enhancement: bool = False,
        phoenix_mode: bool = False,  # Phoenix dataset optimization
        phoenix_frame_sample_rate: int = 200,  # Save every Nth frame in Phoenix mode
        phoenix_json_only: bool = False,  # NEW: JSON-only mode (no frames/videos)
        phoenix_json_name: str = None,  # NEW: Custom JSON filename
        disable_mirroring: bool = False,
        args=None
) -> Optional[Dict[str, Any]]:
    """
    Process video or image sequence for sign language detection including hands, face, and pose landmarks.

    Args:
        phoenix_mode: If True, optimizes for Phoenix dataset processing
        phoenix_json_only: If True, only saves JSON landmark data (no frames, videos, or comparisons)
        phoenix_json_name: Custom name for the JSON file (without .json extension)
    """

    input_path = Path(input_path)

    # Set up frames directory (skip in JSON-only mode)
    if phoenix_mode and phoenix_json_only:
        print("Phoenix JSON-only mode: Skipping frames directory creation")
        frames_dir = None
    else:
        # Always set up frames directory for normal processing
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

    # Set up output video writer (skip for Phoenix JSON-only mode)
    video_writer = None
    if not (phoenix_mode and phoenix_json_only):
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
    else:
        print("Phoenix JSON-only mode: Skipping video output generation")

    frame_count = 0  # Actual frame number (0-based from source)
    processed_frame_count = 0  # Number of frames we've actually processed
    all_frames_data = {}

    print(f"Processing settings:")
    print(f"  - Skip frames: {skip_frames}")
    print(f"  - Save all frames: {save_all_frames}")
    print(f"  - Use enhancement: {use_enhancement}")
    print(f"  - Use full mesh: {use_full_mesh}")
    print(f"  - Enhanced hand tracking: ENABLED")
    print(f"  - Phoenix mode: {'ENABLED' if phoenix_mode else 'DISABLED'}")
    if phoenix_mode:
        print(f"  - Phoenix JSON-only: {'ENABLED' if phoenix_json_only else 'DISABLED'}")
        if phoenix_json_name:
            print(f"  - Custom JSON name: {phoenix_json_name}.json")

    #Initialize enhanced hand tracker instead of regular MediaPipe
    # enhanced_hand_tracker = EnhancedHandTracker(
    #     min_detection_confidence=0.7,
    #     temporal_smoothing_frames=5,
    #     confidence_threshold=0.6,
    #     frame_border_margin=0.1  # 10% of frame width/height considered "near border"
    # )
    enhanced_hand_tracker = EnhancedHandTracker()

    face_tracker = FaceTracker(
        confidence_threshold=0.5,
        temporal_smoothing_frames=3
    ) if extract_face else None

    # Initialize face and pose solutions
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

                    # Determine frame saving path
                    current_frame_path = None  # Default to no saving

                    if phoenix_mode and phoenix_json_only:
                        # JSON-only mode: Never save frames
                        current_frame_path = None
                    elif phoenix_mode and frames_dir is not None:
                        # Regular Phoenix mode: Save frame only every phoenix_frame_sample_rate processed frames
                        if processed_frame_count % phoenix_frame_sample_rate == 0:
                            current_frame_path = frames_dir / f"frame_{current_frame_number:04d}.png"
                    elif frames_dir is not None:
                        # Original logic for non-Phoenix mode
                        if save_all_frames:
                            current_frame_path = frames_dir / f"frame_{current_frame_number:04d}.png"
                        elif processed_frame_count % 10 == 0:
                            current_frame_path = frames_dir / f"frame_{current_frame_number:04d}.png"

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
                        use_enhancement=use_enhancement,
                        phoenix_mode=phoenix_mode,
                        processed_frame_count=processed_frame_count,
                        phoenix_frame_sample_rate=phoenix_frame_sample_rate,
                        phoenix_json_only=phoenix_json_only,
                        disable_mirroring=disable_mirroring,
                        face_tracker=face_tracker
                    )

                    processed_frame_count += 1

                    # Progress reporting
                    progress_interval = 50 if phoenix_mode else 10
                    if processed_frame_count % progress_interval == 0:
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
                        # Determine frame saving path
                        current_frame_path = None  # Default to no saving

                        if phoenix_mode and phoenix_json_only:
                            # JSON-only mode: Never save frames
                            current_frame_path = None
                        elif phoenix_mode and frames_dir is not None:
                            # Regular Phoenix mode: Save frame only every phoenix_frame_sample_rate processed frames
                            if processed_frame_count % phoenix_frame_sample_rate == 0:
                                current_frame_path = frames_dir / f"frame_{frame_count:04d}.png"
                        elif frames_dir is not None:
                            # Original logic for non-Phoenix mode
                            if save_all_frames:
                                current_frame_path = frames_dir / f"frame_{frame_count:04d}.png"
                            elif processed_frame_count % 10 == 0:
                                current_frame_path = frames_dir / f"frame_{frame_count:04d}.png"

                        process_frame_enhanced(
                            frame,
                            frame_count,
                            fps,
                            enhanced_hand_tracker,
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
                            use_enhancement=use_enhancement,
                            phoenix_mode=phoenix_mode,
                            processed_frame_count=processed_frame_count,
                            phoenix_frame_sample_rate=phoenix_frame_sample_rate,
                            phoenix_json_only=phoenix_json_only,
                            disable_mirroring=disable_mirroring,
                            face_tracker=face_tracker
                        )

                        processed_frame_count += 1

                        # Progress reporting
                        progress_interval = 50 if phoenix_mode else 10
                        if processed_frame_count % progress_interval == 0:
                            progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                            print(
                                f"Processed {processed_frame_count} frames (current frame: {frame_count}/{total_frames}, {progress_percent:.1f}%)")

                    frame_count += 1

                # Clean up video capture
                cap.release()

        #TODO: cleanup
        finally:
            # Clean up trackers
            enhanced_hand_tracker.close()

            # Get statistics
            hand_stats = enhanced_hand_tracker.get_statistics()
            disappearance_stats = enhanced_hand_tracker.get_disappearance_statistics()

            face_stats = None
            if face_tracker:
                face_stats = face_tracker.get_disappearance_statistics()

            # Print enhanced tracking statistics
            stats = enhanced_hand_tracker.get_statistics()
            # Print enhanced tracking statistics in unified format
            print(f"\n=== Hand & Face Tracking Statistics ===")
            print(
                f"{'Component':<12} {'Missing Frames':<15} {'Unexpected Disappearances':<25} {'Detection Success Rate':<22}")
            print(f"{'-' * 12} {'-' * 15} {'-' * 25} {'-' * 22}")

            # Left Hand
            left_missing = disappearance_stats['left_hand']['total_missing_frames']
            left_unexpected = disappearance_stats['left_hand']['unexpected_disappearances']
            left_success_rate = f"{disappearance_stats['summary']['left_hand_success_rate']:.1f}%"
            print(f"{'Left Hand':<12} {left_missing:<15} {left_unexpected:<25} {left_success_rate:<22}")

            # Right Hand
            right_missing = disappearance_stats['right_hand']['total_missing_frames']
            right_unexpected = disappearance_stats['right_hand']['unexpected_disappearances']
            right_success_rate = f"{disappearance_stats['summary']['right_hand_success_rate']:.1f}%"
            print(f"{'Right Hand':<12} {right_missing:<15} {right_unexpected:<25} {right_success_rate:<22}")

            # Face - now matching the same format as hands
            if face_stats:
                face_missing = face_stats['face']['total_missing_frames']
                face_unexpected = face_stats['face']['unexpected_disappearances']
                face_success_rate = f"{face_stats['summary']['face_success_rate']:.1f}%"
                print(f"{'Face':<12} {face_missing:<15} {face_unexpected:<25} {face_success_rate:<22}")
            else:
                print(f"{'Face':<12} {'N/A':<15} {'N/A':<25} {'N/A (disabled)':<22}")

            print(f"\n=== Detailed Rates ===")
            print(f"Left hand missing rate: {disappearance_stats['summary']['left_hand_missing_rate']:.1f}%")
            print(f"Right hand missing rate: {disappearance_stats['summary']['right_hand_missing_rate']:.1f}%")
            if face_stats:
                print(f"Face missing rate: {face_stats['summary']['face_missing_rate']:.1f}%")
                # Optional: Add detection filtering success rate for face if needed
                if 'detection_stats' in face_stats and face_stats['detection_stats']['total_detections'] > 0:
                    detection_success = (face_stats['detection_stats']['successful_detections'] /
                                         face_stats['detection_stats']['total_detections']) * 100
                    print(f"Face detection filtering success: {detection_success:.1f}%")

    # Clean up video writer (only if it was created)
    if video_writer is not None:
        video_writer.release()

    # Save metadata - make sure stats is defined
    try:
        stats = enhanced_hand_tracker.get_statistics()
    except:
        stats = {}  # Fallback if tracker is already closed

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
            "enhanced_hand_tracking": True,
            "enhanced_face_tracking": face_tracker is not None,
            "phoenix_mode": phoenix_mode,
            "phoenix_json_only": phoenix_json_only if phoenix_mode else False
        },
        "enhanced_hand_tracking_stats": hand_stats,
        "hand_disappearance_stats": disappearance_stats,
        "face_tracking_stats": face_stats  # This will now have the correct structure
    }

    # Save all frame data to JSON with custom filename
    if phoenix_json_only and phoenix_json_name:
        json_filename = f"{phoenix_json_name}.json"
    else:
        json_filename = "video_landmarks.json"

    json_path = output_dir / json_filename

    try:
        print(f"Saving JSON data to: {json_path}")
        print(f"Number of frames in data: {len(all_frames_data)}")

        with open(json_path, "w") as f:
            json.dump({"metadata": metadata, "frames": all_frames_data}, f, indent=4)
        print(f"Processing complete. Data saved to {json_path}")
        print(f"JSON file size: {json_path.stat().st_size} bytes")
        print(f"Total frames in source: {total_frames}")
        print(f"Total frames processed: {processed_frame_count}")
        if frames_dir is not None:
            print(f"Frames saved to disk: {len(list(frames_dir.glob('frame_*.png')))}")
        else:
            print("JSON-only mode: No frames saved to disk")

        # Count saved frames only if frames directory exists
        if frames_dir and frames_dir.exists():
            frames_saved = len(list(frames_dir.glob('frame_*.png')))
            print(f"Frames saved to disk: {frames_saved}")
        else:
            print(f"Frames saved to disk: 0 (JSON-only mode)")

        # Verify output files exist (only if not in Phoenix JSON-only mode)
        if not (phoenix_mode and phoenix_json_only):
            expected_video_file = output_dir / "annotated_video.mp4"
            if not expected_video_file.exists():
                # Check for fallback .avi file
                expected_video_file = output_dir / "annotated_video.avi"
            if expected_video_file.exists():
                print(f"Output video: {expected_video_file} ({expected_video_file.stat().st_size} bytes)")
            else:
                print(f"WARNING: No output video file found")
        else:
            print("Phoenix JSON-only mode: Video output was skipped")

        return all_frames_data

    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_calibrated_face_landmarks(original_landmarks, translation_offset):
    """Create a new MediaPipe face landmarks structure with calibrated positions"""
    # Create a copy of the original landmarks
    calibrated_landmarks = type(original_landmarks)()
    calibrated_landmarks.CopyFrom(original_landmarks)

    # Apply translation to all landmarks
    for landmark in calibrated_landmarks.landmark:
        landmark.x += translation_offset[0]
        landmark.y += translation_offset[1]
        landmark.z += translation_offset[2]

    return calibrated_landmarks


def process_frame_enhanced(
        frame, actual_frame_number, fps, enhanced_hand_tracker, face_mesh, pose,
        extract_face, extract_pose, all_frames_data,
        annotated_frame_path=None, video_writer=None,
        total_frames=0, skip_frames=1,
        save_all_frames=False,
        use_full_mesh=False,
        use_enhancement=False,
        phoenix_mode=False,
        processed_frame_count=0,
        phoenix_frame_sample_rate=50,
        phoenix_json_only=False,
        disable_mirroring=False,
        face_tracker=None
):
    """
    Enhanced frame processing function that uses the new hand tracker with quality calculation
    """
    try:
        # Apply enhancement if requested
        if use_enhancement:
            # Create comparison save path if we're saving frames
            comparison_save_path = None
            if annotated_frame_path:
                # Create comparisons directory alongside frames directory, not inside it
                frames_dir = annotated_frame_path.parent if annotated_frame_path else Path("temp")
                if annotated_frame_path:
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

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        # Initialize frame data structure - use actual frame number
        frame_data = {
            "frame": actual_frame_number,
            "timestamp": actual_frame_number / fps,
            "hands": {"left_hand": [], "right_hand": []},
            "face": {"all_landmarks": [], "mouth_landmarks": []},
            "pose": {},
            "enhancement_applied": use_enhancement,
            "width": w,
            "height": h,
            "missing_data": {
                "left_hand_missing": True,  # Will be updated below
                "right_hand_missing": True,  # Will be updated below
                "face_missing": True,  # Will be updated below
                "any_missing": True  # Will be calculated
            }
        }

        # STEP 1: Process pose first (needed for calibration references)
        pose_wrists = {"LEFT_WRIST": None, "RIGHT_WRIST": None}
        pose_nose_position = None

        if extract_pose:
            results_pose = pose.process(rgb_frame)
            if results_pose.pose_landmarks:
                pose_data = {}

                for landmark_name in CORE_POSE_LANDMARKS:
                    try:
                        landmark_enum = getattr(mp_pose.PoseLandmark, landmark_name)
                        lm = results_pose.pose_landmarks.landmark[landmark_enum]
                        pose_data[landmark_name] = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": float(lm.visibility)
                        }
                    except AttributeError:
                        continue

                # Store reference positions for calibration
                if "LEFT_WRIST" in pose_data:
                    pose_wrists["LEFT_WRIST"] = np.array([
                        pose_data["LEFT_WRIST"]["x"],
                        pose_data["LEFT_WRIST"]["y"],
                        pose_data["LEFT_WRIST"]["z"]
                    ])

                if "RIGHT_WRIST" in pose_data:
                    pose_wrists["RIGHT_WRIST"] = np.array([
                        pose_data["RIGHT_WRIST"]["x"],
                        pose_data["RIGHT_WRIST"]["y"],
                        pose_data["RIGHT_WRIST"]["z"]
                    ])

                if "NOSE" in pose_data:
                    pose_nose_position = np.array([
                        pose_data["NOSE"]["x"],
                        pose_data["NOSE"]["y"],
                        pose_data["NOSE"]["z"]
                    ])

                frame_data["pose"] = pose_data

                # Draw pose
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
                )

        # STEP 2: Process hands with enhanced tracker
        hands_data = enhanced_hand_tracker.process_frame(rgb_frame)

        # STEP 3: Calibrate hands at MediaPipe object level (BEFORE JSON conversion)
        if pose_wrists["LEFT_WRIST"] is not None or pose_wrists["RIGHT_WRIST"] is not None:
            # Use the existing calibrate_hands_to_wrists function that works with MediaPipe objects
            calibrated_hands_data = calibrate_hands_to_wrists(hands_data, pose_wrists)
        else:
            calibrated_hands_data = hands_data

        # Convert calibrated MediaPipe objects to JSON format
        json_hands_data = enhanced_hand_tracker.get_landmarks_for_json(calibrated_hands_data, frame.shape)
        frame_data["hands"] = json_hands_data

        # STEP 3.5: Calculate hand quality scores (NEW - but don't filter)
        left_hand_quality = enhanced_hand_tracker.calculate_hand_quality(
            calibrated_hands_data.get('left_hand'), 'left_hand'
        )
        right_hand_quality = enhanced_hand_tracker.calculate_hand_quality(
            calibrated_hands_data.get('right_hand'), 'right_hand'
        )

        # Draw hands (use original MediaPipe objects for drawing)
        enhanced_hand_tracker.draw_hands_on_frame(annotated_frame, calibrated_hands_data)

        # STEP 4: Process face
        face_detected = False
        face_quality = {'quality': 0.0, 'context': 'no_detection'}

        if extract_face:
            results_face = face_mesh.process(rgb_frame)

            face_landmarks = None
            face_confidence = 1.0

            # Always process through face tracker, even if no face detected
            if results_face.multi_face_landmarks and len(results_face.multi_face_landmarks) > 0:
                # Face detected by MediaPipe
                face_landmarks = results_face.multi_face_landmarks[0]
                face_confidence = 0.9

                # Apply calibration to MediaPipe object BEFORE processing through tracker
                if pose_nose_position is not None:
                    face_landmarks = calibrate_face_to_nose(face_landmarks, pose_nose_position)
            else:
                # No face detected
                face_landmarks = None
                face_confidence = 0.0

            # CRITICAL: Always process through face tracker for statistics
            if face_tracker:
                tracked_face_result = face_tracker.process_face_detection(face_landmarks, face_confidence)

                if tracked_face_result and tracked_face_result['landmarks']:
                    # Face was detected and passed tracking validation
                    face_landmarks = tracked_face_result['landmarks']  # This is already calibrated
                    face_detected = True
                    face_quality = {'quality': 0.8, 'context': 'detected'}  # Simple quality for now

                    # Extract face data to JSON format from calibrated MediaPipe object
                    face_data = []
                    mouth_data = []

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
                    frame_data["face"]["tracked"] = True
                    frame_data["face"]["detected"] = True
                    frame_data["face"]["calibrated"] = pose_nose_position is not None

                    # Draw face (use original MediaPipe object)
                    if use_full_mesh:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    # Add your existing simplified face drawing code here if needed
                else:
                    # Face not detected or filtered out by tracker
                    frame_data["face"]["tracked"] = True
                    frame_data["face"]["detected"] = False
                    frame_data["face"]["all_landmarks"] = []
                    frame_data["face"]["mouth_landmarks"] = []
            else:
                # Face tracking disabled
                frame_data["face"]["tracked"] = False
                # If face tracking is disabled, we can't determine if face is missing
                # so we'll assume it's not missing for filtering purposes
                face_detected = True
                face_quality = {'quality': 1.0, 'context': 'tracking_disabled'}

        # STEP 4.5: Add quality information to frame data (NEW - preserve all data)
        frame_data["quality_scores"] = {
            'left_hand': left_hand_quality,
            'right_hand': right_hand_quality,
            'face': face_quality,
            'hands_avg_quality': (left_hand_quality['quality'] + right_hand_quality['quality']) / 2,
            'overall_quality': (left_hand_quality['quality'] + right_hand_quality['quality'] + face_quality[
                'quality']) / 3
        }

        # STEP 5: Update missing data detection (ORIGINAL logic - no quality filtering)
        left_hand_data = json_hands_data.get('left_hand', [])
        right_hand_data = json_hands_data.get('right_hand', [])

        # Check if hands are missing using original logic
        def is_hand_data_missing(hand_data):
            if not hand_data:
                return True
            if isinstance(hand_data, dict):
                landmarks = hand_data.get('landmarks', [])
                confidence = hand_data.get('confidence', 0)
                return not landmarks or confidence < 0.1
            elif isinstance(hand_data, list):
                return len(hand_data) == 0
            return True

        frame_data["missing_data"]["left_hand_missing"] = is_hand_data_missing(left_hand_data)
        frame_data["missing_data"]["right_hand_missing"] = is_hand_data_missing(right_hand_data)
        frame_data["missing_data"]["face_missing"] = not face_detected

        # Calculate if any data is missing
        frame_data["missing_data"]["any_missing"] = (
                frame_data["missing_data"]["left_hand_missing"] or
                frame_data["missing_data"]["right_hand_missing"] or
                frame_data["missing_data"]["face_missing"]
        )

        # Debug output for hand positions
        for hand_type in ['left_hand', 'right_hand']:
            hand_data = frame_data["hands"].get(hand_type)
            if hand_data and isinstance(hand_data, dict) and 'landmarks' in hand_data:
                wrist = hand_data['landmarks'][0]
                if DEBUG: print(f"  {hand_type}: wrist at x={wrist['x']:.3f}")
            elif hand_data and isinstance(hand_data, list) and len(hand_data) > 0:
                wrist = hand_data[0]
                if DEBUG: print(f"  {hand_type}: wrist at x={wrist['x']:.3f}")

        # STEP 6: Apply mirroring AFTER calibration
        if not disable_mirroring:
            frame_data = apply_mirroring_to_frame_data(frame_data, pose_wrists, frame.shape)

            for hand_type in ['left_hand', 'right_hand']:
                hand_data = frame_data["hands"].get(hand_type)
                if hand_data and isinstance(hand_data, dict) and 'landmarks' in hand_data:
                    wrist = hand_data['landmarks'][0]
                    if DEBUG: print(f"  {hand_type}: wrist at x={wrist['x']:.3f}")

        # Save frame data using actual frame number as key
        all_frames_data[str(actual_frame_number)] = frame_data

        # Add frame info to image - show actual frame number and enhancement status
        if not (phoenix_mode and phoenix_json_only):
            # Add frame info to image - show actual frame number and enhancement status
            if not phoenix_mode:  # Skip detailed frame info in Phoenix mode for speed
                progress_percent = (actual_frame_number / total_frames) * 100 if total_frames > 0 else 0
                enhancement_text = " (Enhanced)" if use_enhancement else ""
                tracking_text = " | Enhanced Hand Tracking + Quality"
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

            # Write frame to video (skip in Phoenix mode)
            if video_writer and not phoenix_mode:
                video_writer.write(annotated_frame)

    except Exception as e:
        print(f"Error processing frame {actual_frame_number}: {e}")
        import traceback
        traceback.print_exc()
        # Add error frame data to maintain sequence continuity
        error_frame_data = {
            "frame": actual_frame_number,
            "timestamp": actual_frame_number / fps,
            "hands": {"left_hand": [], "right_hand": []},
            "face": {"all_landmarks": [], "mouth_landmarks": []},
            "pose": {},
            "error": True,
            "error_message": str(e),
            "quality_scores": {
                'left_hand': {'quality': 0.0, 'context': 'error'},
                'right_hand': {'quality': 0.0, 'context': 'error'},
                'face': {'quality': 0.0, 'context': 'error'},
                'hands_avg_quality': 0.0,
                'overall_quality': 0.0
            }
        }
        all_frames_data[str(actual_frame_number)] = error_frame_data


def analyze_sequence_quality(json_path: str) -> Dict:
    """Analyze quality scores across a sequence - simplified"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames_data = data.get('frames', {})

    left_qualities = []
    right_qualities = []
    contexts = {'detected': 0, 'no_detection': 0}

    for frame_key, frame_data in frames_data.items():
        quality_scores = frame_data.get('quality_scores', {})

        left_quality = quality_scores.get('left_hand', {})
        right_quality = quality_scores.get('right_hand', {})

        if left_quality.get('quality', 0) > 0:
            left_qualities.append(left_quality['quality'])
            contexts[left_quality.get('context', 'no_detection')] += 1

        if right_quality.get('quality', 0) > 0:
            right_qualities.append(right_quality['quality'])
            contexts[right_quality.get('context', 'no_detection')] += 1

    return {
        'left_hand_stats': {
            'mean_quality': np.mean(left_qualities) if left_qualities else 0,
            'min_quality': np.min(left_qualities) if left_qualities else 0,
            'max_quality': np.max(left_qualities) if left_qualities else 0,
            'detection_rate': len(left_qualities) / len(frames_data) if frames_data else 0
        },
        'right_hand_stats': {
            'mean_quality': np.mean(right_qualities) if right_qualities else 0,
            'min_quality': np.min(right_qualities) if right_qualities else 0,
            'max_quality': np.max(right_qualities) if right_qualities else 0,
            'detection_rate': len(right_qualities) / len(frames_data) if frames_data else 0
        },
        'context_distribution': contexts,
        'total_frames': len(frames_data)
    }


def natural_sort_key(s):
    """
    Sort strings with embedded numbers in natural order.
    For example: frame1.jpg, frame2.jpg, frame10.jpg (instead of frame1.jpg, frame10.jpg, frame2.jpg)
    Enhanced to handle Phoenix dataset frame naming patterns.
    """
    # Handle Phoenix-style frame names like "images-000001.png"
    if 'images-' in s:
        # Extract the number part after 'images-'
        match = re.search(r'images-(\d+)', s)
        if match:
            return int(match.group(1))

    # Original logic for other naming patterns
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def main(json_path):
    quality_stats = analyze_sequence_quality(json_path)
    print(f"Left hand mean quality: {quality_stats['left_hand_stats']['mean_quality']:.2f}")
    print(f"Right hand mean quality: {quality_stats['right_hand_stats']['mean_quality']:.2f}")
    print(f"Context distribution: {quality_stats['context_distribution']}")


if __name__ == "__main__":
    main('./../output_data/phoenix_sample_interpolate_50_quality_calc/01August_2011_Monday_heute_default-6.json')
