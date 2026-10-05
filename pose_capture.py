import cv2
import mediapipe as mp
import json
import time
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

project_directory = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "models" / "pose_landmarker_full.task").exists()
)
output_file = project_directory / "motion_data" / "motion.json"
output_file.parent.mkdir(exist_ok=True)

base_options = python.BaseOptions(
    model_asset_path=str(project_directory / "models" / "pose_landmarker_full.task")
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

motion_data = []

frame_number = 0

print("Starting camera...")
print("Press R to start recording.")
print("Press S to stop recording.")
print("Press Q to quit.")

recording = False

with vision.PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Camera error.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_image, frame_number * 33)
        frame_number += 1

        if results.pose_landmarks:

            for landmark in results.pose_landmarks[0]:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            if recording:

                landmarks = []

                for landmark in results.pose_landmarks[0]:

                    landmarks.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

                motion_data.append({
                    "frame": len(motion_data),
                    "timestamp": time.time(),
                    "landmarks": landmarks
                })

        status = "RECORDING" if recording else "READY"

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if recording else (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(motion_data)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Automatic Character Animation",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):

            print("Recording started.")
            recording = True

        elif key == ord("s"):

            print("Recording stopped.")
            recording = False

            with open(output_file, "w") as f:
                json.dump(motion_data, f, indent=2)

            print(
                f"Saved {len(motion_data)} frames to {output_file}"
            )

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

if motion_data:

    with open(output_file, "w") as f:
        json.dump(motion_data, f, indent=2)

    print(
        f"Motion saved to {output_file}"
    )

print("Program finished.")