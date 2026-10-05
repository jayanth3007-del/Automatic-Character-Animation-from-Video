import json
import numpy as np
from pathlib import Path

PROJECT_DIRECTORY = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "motion_data").exists()
)
INPUT_FILE = PROJECT_DIRECTORY / "motion_data" / "motion.json"
OUTPUT_FILE = PROJECT_DIRECTORY / "motion_data" / "processed_motion.json"


def point(landmarks, index):

    p = landmarks[index]

    return np.array([
        p["x"],
        p["y"],
        p["z"]
    ], dtype=float)


def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return 0.0

    cosine = np.dot(ba, bc) / denominator

    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def process_motion():

    if not INPUT_FILE.exists():

        print("motion.json not found.")
        print("Run pose_capture.py first.")
        return

    with open(INPUT_FILE, "r") as f:

        data = json.load(f)

    processed = []

    for frame_data in data:

        lm = frame_data["landmarks"]

        # MediaPipe landmark indices

        left_shoulder = point(lm, 11)
        right_shoulder = point(lm, 12)

        left_elbow = point(lm, 13)
        right_elbow = point(lm, 14)

        left_wrist = point(lm, 15)
        right_wrist = point(lm, 16)

        left_hip = point(lm, 23)
        right_hip = point(lm, 24)

        left_knee = point(lm, 25)
        right_knee = point(lm, 26)

        left_ankle = point(lm, 27)
        right_ankle = point(lm, 28)

        left_elbow_angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        right_elbow_angle = calculate_angle(
            right_shoulder,
            right_elbow,
            right_wrist
        )

        left_knee_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

        processed.append({

            "frame": frame_data["frame"],

            "left_elbow": left_elbow_angle,

            "right_elbow": right_elbow_angle,

            "left_knee": left_knee_angle,

            "right_knee": right_knee_angle

        })

    with open(OUTPUT_FILE, "w") as f:

        json.dump(processed, f, indent=2)

    print("Motion processing complete.")

    print(
        f"Processed {len(processed)} frames."
    )

    print(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    process_motion()