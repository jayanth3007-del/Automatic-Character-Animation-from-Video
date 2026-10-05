
import cv2
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_DIR / "models" / "pose_landmarker_full.task"

MOTION_DIR = PROJECT_DIR / "motion_data"
VIDEO_DIR = PROJECT_DIR / "input_videos"

MOTION_FILE = MOTION_DIR / "motion.json"

BLENDER_DIR = PROJECT_DIR / "blender"

BLENDER_FILE = BLENDER_DIR / "remy.blend"

ANIMATION_SCRIPT = BLENDER_DIR / "animate_character.py"


# ============================================================
# CREATE FOLDERS
# ============================================================

MOTION_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)
BLENDER_DIR.mkdir(exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_FILE.exists():
    print("\nERROR: pose_landmarker_full.task not found.")
    print(MODEL_FILE)
    input("\nPress Enter to exit...")
    raise SystemExit


if not BLENDER_FILE.exists():
    print("\nERROR: remy.blend not found.")
    print(BLENDER_FILE)
    input("\nPress Enter to exit...")
    raise SystemExit


if not ANIMATION_SCRIPT.exists():
    print("\nERROR: animate_character.py not found.")
    print(ANIMATION_SCRIPT)
    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# FIND BLENDER
# ============================================================

BLENDER_LOCATIONS = [
    Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
]

BLENDER_EXE = None

for blender_path in BLENDER_LOCATIONS:

    if blender_path.exists():

        BLENDER_EXE = blender_path

        break


if BLENDER_EXE is None:

    print("\nERROR: Blender executable was not found.")

    input("\nPress Enter to exit...")

    raise SystemExit


# ============================================================
# IMPORT MEDIAPIPE
# ============================================================

print("\nLoading MediaPipe...")

try:

    import mediapipe as mp

    from mediapipe.tasks import python

    from mediapipe.tasks.python import vision

except Exception as error:

    print("\nERROR while loading MediaPipe:")
    print(error)

    print("\nThis is a MediaPipe installation problem.")

    input("\nPress Enter to exit...")

    raise SystemExit


# ============================================================
# CREATE POSE LANDMARKER
# ============================================================

print("Loading pose model...")

try:

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_FILE)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.PoseLandmarker.create_from_options(
        options
    )

except Exception as error:

    print("\nERROR while creating MediaPipe Pose Landmarker:")
    print(error)

    input("\nPress Enter to exit...")

    raise SystemExit


# ============================================================
# OPEN CAMERA
# ============================================================

print("\nOpening camera...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


if not cap.isOpened():

    cap.release()

    print("DirectShow camera opening failed.")
    print("Trying normal camera mode...")

    cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("\nERROR: Camera could not be opened.")

    landmarker.close()

    input("\nPress Enter to exit...")

    raise SystemExit


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)

try:

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

except Exception:
    pass


actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print(
    f"\nCamera resolution: "
    f"{actual_width} x {actual_height}"
)


# ============================================================
# CREATE VIDEO FILE
# ============================================================

time_string = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

video_file = (
    VIDEO_DIR /
    f"input_motion_{time_string}.mp4"
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


video_writer = cv2.VideoWriter(
    str(video_file),
    fourcc,
    30.0,
    (
        actual_width,
        actual_height
    )
)


# ============================================================
# VARIABLES
# ============================================================

motion_data = []

recording = False

frame_number = 0

start_time = None


# ============================================================
# READY
# ============================================================

print("\n")
print("=" * 60)
print("CAMERA READY")
print("=" * 60)

print("\nR = Start recording")
print("S = Stop recording")
print("Q = Quit")

print("\nPress R to start recording.")


# ============================================================
# CAMERA LOOP
# ============================================================

try:

    while True:

        success, frame = cap.read()


        # ----------------------------------------------------
        # CAMERA READ ERROR
        # ----------------------------------------------------

        if not success or frame is None:

            print(
                "\nWARNING: Camera frame could not be read."
            )

            time.sleep(0.05)

            continue


        # ----------------------------------------------------
        # RESIZE IF REQUIRED
        # ----------------------------------------------------

        if (
            frame.shape[1] != actual_width
            or
            frame.shape[0] != actual_height
        ):

            frame = cv2.resize(
                frame,
                (
                    actual_width,
                    actual_height
                )
            )


        # ----------------------------------------------------
        # MIRROR DISPLAY
        # ----------------------------------------------------

        display_frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # RECORDING
        # ====================================================

        if recording:

            current_time = time.monotonic()

            elapsed_seconds = (
                current_time -
                start_time
            )

            timestamp_ms = int(
                elapsed_seconds * 1000
            )


            # ------------------------------------------------
            # MAKE TIMESTAMP INCREASE
            # ------------------------------------------------

            if len(motion_data) > 0:

                previous_timestamp = (
                    motion_data[-1]["timestamp"]
                )

                if timestamp_ms <= previous_timestamp:

                    timestamp_ms = (
                        previous_timestamp + 1
                    )


            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # MEDIAPIPE IMAGE
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # ------------------------------------------------
            # POSE DETECTION
            # ------------------------------------------------

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )


            # ------------------------------------------------
            # SAVE BODY LANDMARKS
            # ------------------------------------------------

            if result.pose_landmarks:

                pose_landmarks = (
                    result.pose_landmarks[0]
                )


                landmarks = []


                for landmark in pose_landmarks:

                    landmarks.append(
                        {
                            "x": float(
                                landmark.x
                            ),

                            "y": float(
                                landmark.y
                            ),

                            "z": float(
                                landmark.z
                            ),

                            "visibility": float(
                                getattr(
                                    landmark,
                                    "visibility",
                                    1.0
                                )
                            )
                        }
                    )


                motion_data.append(
                    {
                        "frame": frame_number,

                        "timestamp": timestamp_ms,

                        "landmarks": landmarks
                    }
                )


                frame_number += 1


            # ------------------------------------------------
            # SAVE VIDEO
            # ------------------------------------------------

            if video_writer.isOpened():

                video_writer.write(
                    frame
                )


            # ------------------------------------------------
            # RECORDING DISPLAY
            # ------------------------------------------------

            cv2.circle(
                display_frame,
                (30, 30),
                12,
                (0, 0, 255),
                -1
            )


            cv2.putText(
                display_frame,
                "RECORDING",
                (55, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


            cv2.putText(
                display_frame,
                f"Frames: {len(motion_data)}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        else:

            cv2.putText(
                display_frame,
                "Press R to record",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Automatic Character Animation - Camera",
            display_frame
        )


        # ====================================================
        # KEY
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ====================================================
        # START
        # ====================================================

        if key == ord("r") or key == ord("R"):

            if not recording:

                motion_data = []

                frame_number = 0

                start_time = time.monotonic()

                recording = True

                print("\n")
                print("=" * 60)
                print("RECORDING STARTED")
                print("=" * 60)

                print(
                    "\nMove your body now."
                )

                print(
                    "Press S when finished."
                )


        # ====================================================
        # STOP
        # ====================================================

        elif key == ord("s") or key == ord("S"):

            if recording:

                recording = False

                print("\n")
                print("=" * 60)
                print("RECORDING STOPPED")
                print("=" * 60)

                print(
                    f"\nCaptured "
                    f"{len(motion_data)} frames."
                )

                break


        # ====================================================
        # QUIT
        # ====================================================

        elif key == ord("q") or key == ord("Q"):

            recording = False

            motion_data = []

            print("\nProgram cancelled.")

            break


# ============================================================
# CLEANUP CAMERA
# ============================================================

finally:

    cap.release()

    if video_writer.isOpened():

        video_writer.release()

    cv2.destroyAllWindows()

    landmarker.close()


# ============================================================
# CHECK RECORDING
# ============================================================

if len(motion_data) == 0:

    print("\nNo motion was recorded.")

    input("\nPress Enter to exit...")

    raise SystemExit


# ============================================================
# SAVE MOTION DATA
# ============================================================

motion_output = {

    "total_frames": len(
        motion_data
    ),

    "landmark_count": 33,

    "fps": 30,

    "frames": motion_data
}


with open(
    MOTION_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        motion_output,
        file,
        indent=2
    )


# ============================================================
# RESULT
# ============================================================

print("\n")
print("=" * 60)
print("MOTION DATA SAVED")
print("=" * 60)

print(
    f"\nFrames captured: "
    f"{len(motion_data)}"
)

print(
    f"\nMotion file:"
)

print(
    MOTION_FILE
)

print(
    f"\nVideo file:"
)

print(
    video_file
)


# ============================================================
# ASK FOR BLENDER
# ============================================================

choice = input(
    "\nCan I proceed to character animation? (Y/N): "
).strip().lower()


if choice != "y":

    print(
        "\nAnimation cancelled."
    )

    print(
        "Your motion data has been saved."
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


# ============================================================
# START BLENDER
# ============================================================

print("\n")
print("=" * 60)
print("STARTING BLENDER")
print("=" * 60)

print("\nPlease wait...")


try:

    blender_process = subprocess.Popen(
        [
            str(BLENDER_EXE),

            str(BLENDER_FILE),

            "--python",

            str(ANIMATION_SCRIPT)
        ]
    )

except Exception as error:

    print(
        "\nERROR: Could not start Blender."
    )

    print(error)

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


print("\nBlender has been started.")

print(
    "\nWhen Blender opens:"
)

print(
    "1. Wait for Remy to load."
)

print(
    "2. Press SPACEBAR."
)

print(
    "3. Watch the animation."
)


input(
    "\nPress Enter to close this window..."
)
