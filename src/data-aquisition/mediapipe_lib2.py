import cv2
import os
import json
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Mouth landmark indices from MediaPipe FaceMesh
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]

# For static images:
IMAGE_FILES = ["../../data/stockimg.png"]
BG_COLOR = (192, 192, 192)  # Gray background
landmarks_data = {}  # Dictionary to store landmark data

with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose, \
        mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.7) as face_mesh, \
        mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue

        image = cv2.imread(file)
        image_height, image_width, _ = image.shape

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process face, pose, and hands separately
        results_pose = pose.process(rgb_image)
        results_face = face_mesh.process(rgb_image)
        results_hands = hands.process(rgb_image)

        print(f"Processing: {file}")

        annotated_image = image.copy()
        image_data = {"face": {}, "pose": {}, "hands": {}}

        # Extract face landmarks & mouth landmarks
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                face_data = []
                mouth_data = []

                for i, lm in enumerate(face_landmarks.landmark):
                    point = {"x": lm.x, "y": lm.y, "z": lm.z}
                    face_data.append(point)

                    # Save only mouth landmarks separately
                    if i in MOUTH_LANDMARKS:
                        mouth_data.append(point)

                image_data["face"]["all_landmarks"] = face_data
                image_data["face"]["mouth_landmarks"] = mouth_data

        # Extract pose landmarks
        if results_pose.pose_landmarks:
            pose_data = {}
            for landmark in mp_pose.PoseLandmark:
                lm = results_pose.pose_landmarks.landmark[landmark]
                pose_data[landmark.name] = {"x": lm.x, "y": lm.y, "z": lm.z}
            image_data["pose"] = pose_data

        # Extract hand landmarks
        if results_hands.multi_hand_landmarks:
            hands_data = {"left_hand": [], "right_hand": []}
            for hand_landmarks in results_hands.multi_hand_landmarks:
                hand_points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]

                # Assign to left or right hand (heuristic: x < 0.5 is left, x > 0.5 is right)
                if hand_landmarks.landmark[0].x < 0.5:
                    hands_data["left_hand"] = hand_points
                else:
                    hands_data["right_hand"] = hand_points

            image_data["hands"] = hands_data

        # Save extracted data
        landmarks_data[file] = image_data

        # Draw annotations
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Draw mouth landmarks
                for i in MOUTH_LANDMARKS:
                    lm = face_landmarks.landmark[i]
                    x, y = int(lm.x * image_width), int(lm.y * image_height)
                    cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Save the annotated image
        os.makedirs('./tmp', exist_ok=True)
        output_path = f'./tmp/annotated_image_{idx}.png'
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

# Save landmark data to JSON
json_output_path = "./tmp/landmarks_data.json"
with open(json_output_path, "w") as json_file:
    json.dump(landmarks_data, json_file, indent=4)

print(f"Landmark data saved to {json_output_path}")
