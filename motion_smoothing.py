import json
import numpy as np
from pathlib import Path

PROJECT_DIRECTORY = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "motion_data").exists()
)
INPUT_FILE = PROJECT_DIRECTORY / "motion_data" / "processed_motion.json"

OUTPUT_FILE = PROJECT_DIRECTORY / "motion_data" / "smoothed_motion.json"

WINDOW = 5


def smooth(values):

    if len(values) < WINDOW:
        return values

    result = []

    for i in range(len(values)):

        start = max(0, i - WINDOW // 2)

        end = min(
            len(values),
            i + WINDOW // 2 + 1
        )

        result.append(
            float(np.mean(values[start:end]))
        )

    return result


with open(INPUT_FILE, "r") as f:

    data = json.load(f)


joint_names = [
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee"
]


for joint in joint_names:

    values = [
        frame[joint]
        for frame in data
    ]

    smoothed = smooth(values)

    for i in range(len(data)):

        data[i][joint] = smoothed[i]


with open(OUTPUT_FILE, "w") as f:

    json.dump(data, f, indent=2)


print("Temporal smoothing complete.")

print(
    f"Saved to {OUTPUT_FILE}"
)