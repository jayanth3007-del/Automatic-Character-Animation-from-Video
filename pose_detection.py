from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def find_model_path():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "models" / "pose_landmarker_full.task"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find pose_landmarker_full.task in the project models folder."
    )


model_path = find_model_path()
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = None
for camera_index in (0, 1, 2):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Using camera index {camera_index}")
        break
    cap.release()

if cap is None or not cap.isOpened():
    raise RuntimeError("Could not open any available camera")

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_number = 0

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read camera")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = frame_number * 33
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    for landmark in landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            cv2.imshow("Automatic Character Animation - Pose Detection", frame)
            frame_number += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()