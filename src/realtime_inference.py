#!/usr/bin/env python3
"""
Real-Time LSTM Inference for Sign Language Recognition

This script demonstrates real-time sign language recognition using the trained LSTM model.
It processes video frames as they arrive, maintaining hidden state between frames.

Usage:
    python realtime_inference.py --model_path ./models/lstm_sign2gloss.pth --camera_id 0
    python realtime_inference.py --model_path ./models/lstm_sign2gloss.pth --video_path ./test_video.mp4
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
import time
import pickle
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import SignLanguageLSTM, ModelConfig
from preprocessJsons import SignLanguagePreprocessor, PreprocessingConfig
from processing import EnhancedHandTracker, calibrate_hands_to_wrists, calibrate_face_to_nose
import mediapipe as mp


class RealTimeSignLanguageRecognizer:
    """Real-time sign language recognition system"""

    def __init__(self, model_path: str, vocab_path: str, config_path: str = None):
        """
        Initialize real-time recognizer

        Args:
            model_path: Path to trained LSTM model
            vocab_path: Path to vocabulary file
            config_path: Optional path to config file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and configuration
        self._load_model(model_path, config_path)
        self._load_vocabulary(vocab_path)

        # Initialize preprocessor
        preprocess_config = PreprocessingConfig(
            max_sequence_length=self.model_config.max_sequence_length,
            output_format="tensor",
            device="cpu"  # Keep preprocessing on CPU for real-time
        )
        self.preprocessor = SignLanguagePreprocessor(preprocess_config)

        # Initialize MediaPipe components
        self._init_mediapipe()

        # Real-time processing state
        self.hidden_state = None
        self.frame_buffer = deque(maxlen=30)  # Buffer for smoothing
        self.prediction_buffer = deque(maxlen=10)  # Buffer predictions
        self.confidence_threshold = 0.7
        self.prediction_smoothing = True

        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0
        self.fps_counter = deque(maxlen=30)

        print(f"Real-time recognizer initialized on {self.device}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Model: {self.model_config.hidden_size}H x {self.model_config.num_layers}L LSTM")

    def _load_model(self, model_path: str, config_path: str = None):
        """Load trained model and configuration"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_config = checkpoint['config']
        vocab_size = checkpoint['vocab_size']

        # Initialize model
        self.model = SignLanguageLSTM(self.model_config, vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")

    def _load_vocabulary(self, vocab_path: str):
        """Load vocabulary mapping"""
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Create reverse mapping
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}

        print(f"Vocabulary loaded: {len(self.vocab)} tokens")

    def _init_mediapipe(self):
        """Initialize MediaPipe components"""
        # Enhanced hand tracker
        self.hand_tracker = EnhancedHandTracker(
            min_detection_confidence=0.7,
            temporal_smoothing_frames=3,  # Reduced for real-time
            confidence_threshold=0.6
        )

        # Face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )

        # Pose detection
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("MediaPipe components initialized")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return recognition results

        Args:
            frame: Input frame (BGR format)

        Returns:
            Dictionary with prediction results and metadata
        """
        start_time = time.time()

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Extract landmarks using existing processing pipeline
        frame_data = self._extract_landmarks(rgb_frame, h, w)

        # Extract features using preprocessor
        if frame_data:
            features = self.preprocessor.extract_frame_features(frame_data)
            features = self.preprocessor.normalize_coordinates(features)

            # Convert to tensor
            features_tensor = torch.from_numpy(features).float().to(self.device)

            # Real-time prediction
            result = self.model.predict_realtime(features_tensor, self.hidden_state)

            # Update hidden state
            self.hidden_state = result['hidden_state']

            # Decode prediction
            prediction_id = result['prediction']
            confidence = result['confidence']

            # Get token text
            prediction_token = self.id_to_vocab.get(prediction_id, '<UNK>')

            # Apply smoothing if enabled
            if self.prediction_smoothing:
                self.prediction_buffer.append({
                    'token': prediction_token,
                    'confidence': confidence,
                    'id': prediction_id
                })

                # Get smoothed prediction
                smoothed_prediction = self._get_smoothed_prediction()
            else:
                smoothed_prediction = {
                    'token': prediction_token,
                    'confidence': confidence,
                    'id': prediction_id
                }
        else:
            # No landmarks detected
            smoothed_prediction = {
                'token': '<NO_GESTURE>',
                'confidence': 0.0,
                'id': -1
            }

        # Calculate timing
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1

        # Update FPS counter
        self.fps_counter.append(time.time())

        return {
            'prediction': smoothed_prediction,
            'inference_time': inference_time,
            'frame_count': self.frame_count,
            'has_landmarks': frame_data is not None
        }

    def _extract_landmarks(self, rgb_frame: np.ndarray, h: int, w: int) -> Optional[Dict]:
        """Extract landmarks from frame using MediaPipe"""
        try:
            frame_data = {
                "hands": {"left_hand": [], "right_hand": []},
                "face": {"all_landmarks": []},
                "pose": {}
            }

            # Process pose first (needed for calibration)
            pose_wrists = {"LEFT_WRIST": None, "RIGHT_WRIST": None}
            pose_nose_position = None

            results_pose = self.pose.process(rgb_frame)
            if results_pose.pose_landmarks:
                pose_data = {}

                # Extract core pose landmarks
                pose_landmark_names = [
                    'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP'
                ]

                for landmark_name in pose_landmark_names:
                    try:
                        landmark_enum = getattr(mp.solutions.pose.PoseLandmark, landmark_name)
                        lm = results_pose.pose_landmarks.landmark[landmark_enum]
                        pose_data[landmark_name] = {
                            "x": lm.x, "y": lm.y, "z": lm.z,
                            "visibility": float(lm.visibility)
                        }
                    except AttributeError:
                        continue

                # Store reference positions
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

            # Process hands
            hands_data = self.hand_tracker.process_frame(rgb_frame)

            # Calibrate hands
            if pose_wrists["LEFT_WRIST"] is not None or pose_wrists["RIGHT_WRIST"] is not None:
                calibrated_hands_data = calibrate_hands_to_wrists(hands_data, pose_wrists)
            else:
                calibrated_hands_data = hands_data

            # Convert to JSON format
            json_hands_data = self.hand_tracker.get_landmarks_for_json(calibrated_hands_data, (h, w, 3))
            frame_data["hands"] = json_hands_data

            # Process face
            results_face = self.face_mesh.process(rgb_frame)
            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]

                # Calibrate face
                if pose_nose_position is not None:
                    face_landmarks = calibrate_face_to_nose(face_landmarks, pose_nose_position)

                # Extract face data
                face_data = []
                for i, lm in enumerate(face_landmarks.landmark):
                    if i % 2 == 0:  # Sample every 2nd landmark
                        face_data.append({
                            "x": lm.x, "y": lm.y, "z": lm.z
                        })

                frame_data["face"]["all_landmarks"] = face_data

            # Check if we have any meaningful data
            has_hands = any(frame_data["hands"][hand] for hand in ["left_hand", "right_hand"])
            has_face = bool(frame_data["face"]["all_landmarks"])
            has_pose = bool(frame_data["pose"])

            if has_hands or has_face or has_pose:
                return frame_data
            else:
                return None

        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None

    def _get_smoothed_prediction(self) -> Dict:
        """Get smoothed prediction from buffer"""
        if not self.prediction_buffer:
            return {'token': '<NO_GESTURE>', 'confidence': 0.0, 'id': -1}

        # Count token frequencies
        token_counts = {}
        confidence_sums = {}

        for pred in self.prediction_buffer:
            token = pred['token']
            confidence = pred['confidence']

            if token not in token_counts:
                token_counts[token] = 0
                confidence_sums[token] = 0.0

            token_counts[token] += 1
            confidence_sums[token] += confidence

        # Find most frequent token with high confidence
        best_token = None
        best_score = 0

        for token, count in token_counts.items():
            avg_confidence = confidence_sums[token] / count
            frequency_score = count / len(self.prediction_buffer)
            combined_score = avg_confidence * frequency_score

            if combined_score > best_score and avg_confidence > self.confidence_threshold:
                best_score = combined_score
                best_token = token

        if best_token:
            return {
                'token': best_token,
                'confidence': confidence_sums[best_token] / token_counts[best_token],
                'id': self.vocab.get(best_token, -1)
            }
        else:
            return {'token': '<LOW_CONFIDENCE>', 'confidence': 0.0, 'id': -1}

    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.fps_counter) < 2:
            return 0.0

        time_span = self.fps_counter[-1] - self.fps_counter[0]
        if time_span > 0:
            return (len(self.fps_counter) - 1) / time_span
        return 0.0

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        avg_inference_time = self.total_inference_time / max(1, self.frame_count)

        return {
            'frames_processed': self.frame_count,
            'avg_inference_time': avg_inference_time,
            'current_fps': self.get_fps(),
            'total_time': self.total_inference_time
        }

    def reset_state(self):
        """Reset recognition state"""
        self.hidden_state = None
        self.prediction_buffer.clear()
        print("Recognition state reset")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'hand_tracker'):
            self.hand_tracker.close()


def draw_results(frame: np.ndarray, result: Dict, stats: Dict) -> np.ndarray:
    """Draw recognition results on frame"""
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]

    # Draw prediction
    prediction = result['prediction']
    token = prediction['token']
    confidence = prediction['confidence']

    # Color based on confidence
    if confidence > 0.8:
        color = (0, 255, 0)  # Green
    elif confidence > 0.6:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red

    # Draw main prediction
    text = f"Sign: {token}"
    cv2.putText(annotated_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Draw confidence
    conf_text = f"Confidence: {confidence:.2f}"
    cv2.putText(annotated_frame, conf_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw statistics
    fps_text = f"FPS: {stats['current_fps']:.1f}"
    cv2.putText(annotated_frame, fps_text, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    inference_text = f"Inference: {result['inference_time'] * 1000:.1f}ms"
    cv2.putText(annotated_frame, inference_text, (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw landmark detection status
    landmark_status = "Landmarks: OK" if result['has_landmarks'] else "Landmarks: NONE"
    status_color = (0, 255, 0) if result['has_landmarks'] else (0, 0, 255)
    cv2.putText(annotated_frame, landmark_status, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    return annotated_frame


def main():
    """Main real-time recognition function"""
    parser = argparse.ArgumentParser(description='Real-time sign language recognition')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='Path to vocabulary')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID for webcam')
    parser.add_argument('--video_path', type=str, help='Path to video file (alternative to camera)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--no_smoothing', action='store_true', help='Disable prediction smoothing')
    parser.add_argument('--save_video', type=str, help='Save annotated video to file')

    args = parser.parse_args()

    # Initialize recognizer
    try:
        recognizer = RealTimeSignLanguageRecognizer(
            model_path=args.model_path,
            vocab_path=args.vocab_path
        )

        recognizer.confidence_threshold = args.confidence_threshold
        recognizer.prediction_smoothing = not args.no_smoothing

    except Exception as e:
        print(f"Error initializing recognizer: {e}")
        return

    # Initialize video source
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
        print(f"Processing video: {args.video_path}")
    else:
        cap = cv2.VideoCapture(args.camera_id)
        print(f"Using camera: {args.camera_id}")

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Set up video writer if saving
    video_writer = None
    if args.save_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
        print(f"Saving video to: {args.save_video}")

    print("\nReal-time recognition started!")
    print("Controls:")
    print("  'r' - Reset recognition state")
    print("  'q' - Quit")
    print("  'h' - Toggle help display")
    print()

    show_help = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or camera disconnected")
                break

            # Process frame
            result = recognizer.process_frame(frame)

            # Get statistics
            stats = recognizer.get_statistics()

            # Draw results
            annotated_frame = draw_results(frame, result, stats)

            # Draw help if enabled
            if show_help:
                cv2.putText(annotated_frame, "Controls: 'r'=reset, 'q'=quit, 'h'=help",
                            (10, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Save frame if needed
            if video_writer:
                video_writer.write(annotated_frame)

            # Display frame
            cv2.imshow('Real-Time Sign Language Recognition', annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recognizer.reset_state()
                print("Recognition state reset")
            elif key == ord('h'):
                show_help = not show_help

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # Print final statistics
        final_stats = recognizer.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Frames processed: {final_stats['frames_processed']}")
        print(f"  Average inference time: {final_stats['avg_inference_time'] * 1000:.1f}ms")
        print(f"  Average FPS: {final_stats['current_fps']:.1f}")


if __name__ == "__main__":
    main()