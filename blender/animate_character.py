import bpy
import json
import math
from pathlib import Path
from mathutils import Vector

# ============================================================
# AUTOMATIC CHARACTER ANIMATION
# Blender 5.x / Mixamo Remy
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Senthil kumar\Desktop\Automatic_Character_Animation")
MOTION_FILE = PROJECT_DIR / "motion_data" / "motion.json"
OUTPUT_FILE = PROJECT_DIR / "blender" / "remy_animated.blend"

FPS = 30
SMOOTH_WINDOW = 3

# ------------------------------------------------------------
# MediaPipe landmark numbers
# ------------------------------------------------------------

NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ------------------------------------------------------------
# Mixamo bone names
# ------------------------------------------------------------

BONES = {
    "hips": "mixamorig:Hips",

    "spine": "mixamorig:Spine",
    "spine1": "mixamorig:Spine1",
    "spine2": "mixamorig:Spine2",

    "neck": "mixamorig:Neck",
    "head": "mixamorig:Head",

    "left_shoulder": "mixamorig:LeftShoulder",
    "left_arm": "mixamorig:LeftArm",
    "left_forearm": "mixamorig:LeftForeArm",
    "left_hand": "mixamorig:LeftHand",

    "right_shoulder": "mixamorig:RightShoulder",
    "right_arm": "mixamorig:RightArm",
    "right_forearm": "mixamorig:RightForeArm",
    "right_hand": "mixamorig:RightHand",

    "left_up_leg": "mixamorig:LeftUpLeg",
    "left_leg": "mixamorig:LeftLeg",
    "left_foot": "mixamorig:LeftFoot",

    "right_up_leg": "mixamorig:RightUpLeg",
    "right_leg": "mixamorig:RightLeg",
    "right_foot": "mixamorig:RightFoot",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def get_armature():
    arm = bpy.data.objects.get("Armature")

    if arm is None:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                arm = obj
                break

    if arm is None:
        raise RuntimeError("Armature not found.")

    return arm


def get_bone(arm, name):
    bone = arm.pose.bones.get(name)

    if bone is None:
        raise RuntimeError("Bone not found: " + name)

    return bone


def clamp(v, a, b):
    return max(a, min(b, v))


# ============================================================
# LOAD MOTION JSON
# ============================================================

def load_motion():

    if not MOTION_FILE.exists():
        raise RuntimeError(
            "motion.json not found:\n" + str(MOTION_FILE)
        )

    with open(MOTION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = []

    if isinstance(data, list):
        frames = data

    elif isinstance(data, dict):

        for key in [
            "frames",
            "motion",
            "landmarks",
            "poses",
            "data"
        ]:
            if key in data and isinstance(data[key], list):
                frames = data[key]
                break

    if not frames:
        raise RuntimeError("Could not find frames in motion.json")

    clean = []

    for frame in frames:

        if isinstance(frame, dict):

            if "landmarks" in frame:
                lm = frame["landmarks"]

            elif "pose_landmarks" in frame:
                lm = frame["pose_landmarks"]

            else:
                continue

        else:
            lm = frame

        if not isinstance(lm, list):
            continue

        if len(lm) < 29:
            continue

        converted = []

        for p in lm:

            if isinstance(p, dict):
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
                z = float(p.get("z", 0.0))

            else:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2]) if len(p) > 2 else 0.0

            converted.append(Vector((x, y, z)))

        clean.append(converted)

    if len(clean) < 2:
        raise RuntimeError("Not enough valid motion frames.")

    print("Loaded motion frames:", len(clean))

    return clean


# ============================================================
# SMOOTH MOTION
# ============================================================

def smooth_motion(frames, window=3):

    if window <= 1:
        return frames

    result = []

    half = window // 2

    for i in range(len(frames)):

        new_frame = []

        start = max(0, i - half)
        end = min(len(frames), i + half + 1)

        for j in range(len(frames[i])):

            total = Vector((0, 0, 0))
            count = 0

            for k in range(start, end):
                total += frames[k][j]
                count += 1

            new_frame.append(total / count)

        result.append(new_frame)

    return result


# ============================================================
# DISTANCE
# ============================================================

def dist(a, b):
    return (a - b).length


# ============================================================
# AVERAGE POINT
# ============================================================

def midpoint(a, b):
    return (a + b) / 2.0


# ============================================================
# GET REST BONE WORLD POSITIONS
# ============================================================

def world_bone_head(arm, bone_name):

    bone = arm.data.bones.get(bone_name)

    if bone is None:
        return Vector((0, 0, 0))

    return arm.matrix_world @ bone.head_local


def world_bone_tail(arm, bone_name):

    bone = arm.data.bones.get(bone_name)

    if bone is None:
        return Vector((0, 0, 0))

    return arm.matrix_world @ bone.tail_local


# ============================================================
# CREATE EMPTY
# ============================================================

def create_empty(name):

    old = bpy.data.objects.get(name)

    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    obj = bpy.data.objects.new(name, None)

    bpy.context.collection.objects.link(obj)

    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = 0.15

    return obj


# ============================================================
# REMOVE OLD TARGETS
# ============================================================

def remove_old_targets():

    names = [
        "MP_RightHand_Target",
        "MP_LeftHand_Target",
        "MP_RightFoot_Target",
        "MP_LeftFoot_Target",
        "MP_RightKnee_Target",
        "MP_LeftKnee_Target",
    ]

    for name in names:

        obj = bpy.data.objects.get(name)

        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)


# ============================================================
# REMOVE OLD IK
# ============================================================

def remove_old_ik(arm):

    for pb in arm.pose.bones:

        for c in list(pb.constraints):

            if c.name.startswith("AUTO_MP"):
                pb.constraints.remove(c)


# ============================================================
# RESET ARMATURE
# ============================================================

def reset_armature(arm):

    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)

    bpy.ops.object.mode_set(mode='POSE')

    for pb in arm.pose.bones:

        pb.location = (0, 0, 0)

        if pb.rotation_mode != 'QUATERNION':
            pb.rotation_mode = 'QUATERNION'

        pb.rotation_quaternion = (1, 0, 0, 0)

        pb.scale = (1, 1, 1)

    bpy.ops.object.mode_set(mode='OBJECT')

    arm.animation_data_clear()


# ============================================================
# FIND CHARACTER MESHES
# ============================================================

def prepare_scene(arm):

    # Keep armature itself exactly where the original
    # Remy scene placed it.

    # Never rotate the armature object.
    arm.rotation_mode = 'XYZ'

    # Do NOT change:
    # arm.location
    # arm.rotation_euler
    # arm.scale

    # Make all meshes visible.
    for obj in bpy.context.scene.objects:

        if obj.type == 'MESH':
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_set(False)


# ============================================================
# CALCULATE CHARACTER SCALE
# ============================================================

def calculate_scale(arm, motion):

    first = motion[0]

    mp_shoulders = dist(
        first[LEFT_SHOULDER],
        first[RIGHT_SHOULDER]
    )

    mp_body = dist(
        midpoint(first[LEFT_SHOULDER], first[RIGHT_SHOULDER]),
        midpoint(first[LEFT_HIP], first[RIGHT_HIP])
    )

    char_left_shoulder = world_bone_head(
        arm,
        BONES["left_arm"]
    )

    char_right_shoulder = world_bone_head(
        arm,
        BONES["right_arm"]
    )

    char_shoulders = dist(
        char_left_shoulder,
        char_right_shoulder
    )

    char_hips = midpoint(
        world_bone_head(arm, BONES["left_up_leg"]),
        world_bone_head(arm, BONES["right_up_leg"])
    )

    char_shoulders_center = midpoint(
        char_left_shoulder,
        char_right_shoulder
    )

    char_body = dist(
        char_shoulders_center,
        char_hips
    )

    if mp_shoulders < 0.01:
        shoulder_scale = 1.0
    else:
        shoulder_scale = char_shoulders / mp_shoulders

    if mp_body < 0.01:
        body_scale = 1.0
    else:
        body_scale = char_body / mp_body

    scale = (shoulder_scale + body_scale) / 2.0

    scale = clamp(scale, 0.5, 10.0)

    print("Character scale:", scale)

    return scale


# ============================================================
# MEDIA PIPE -> CHARACTER SPACE
#
# IMPORTANT:
# We use RELATIVE movement.
#
# This prevents:
# - automatic forward bending
# - automatic sideways rotation
# - character drifting away
# ============================================================

def convert_relative(mp_point, mp_reference, rest_point, scale):

    dx = mp_point.x - mp_reference.x
    dy = mp_point.y - mp_reference.y
    dz = mp_point.z - mp_reference.z

    # MediaPipe:
    # x = right
    # y = down
    # z = depth
    #
    # Blender:
    # x = right
    # y = depth
    # z = up

    result = Vector((
        dx * scale,
        -dz * scale,
        -dy * scale
    ))

    return rest_point + result


# ============================================================
# CREATE IK TARGET
# ============================================================

def add_ik_target(
    arm,
    bone_name,
    target,
    chain_length
):

    pb = get_bone(arm, bone_name)

    c = pb.constraints.new('IK')

    c.name = "AUTO_MP_" + bone_name

    c.target = target

    c.chain_count = chain_length

    c.use_stretch = False

    return c


# ============================================================
# CREATE MOTION TARGETS
# ============================================================

def create_targets():

    return {
        "right_hand": create_empty("MP_RightHand_Target"),
        "left_hand": create_empty("MP_LeftHand_Target"),
        "right_foot": create_empty("MP_RightFoot_Target"),
        "left_foot": create_empty("MP_LeftFoot_Target"),
    }


# ============================================================
# SET TARGET LOCATION
# ============================================================

def set_target(obj, position):

    obj.location = position


# ============================================================
# MAIN ANIMATION
# ============================================================

def animate():

    print("")
    print("======================================")
    print(" AUTOMATIC CHARACTER ANIMATION")
    print("======================================")

    motion = load_motion()

    motion = smooth_motion(
        motion,
        SMOOTH_WINDOW
    )

    arm = get_armature()

    print("Armature:", arm.name)

    # --------------------------------------------------------
    # IMPORTANT:
    # Save original object transform.
    # --------------------------------------------------------

    original_location = arm.location.copy()
    original_rotation = arm.rotation_euler.copy()
    original_scale = arm.scale.copy()

    # --------------------------------------------------------
    # Clean previous generated animation
    # --------------------------------------------------------

    remove_old_targets()

    remove_old_ik(arm)

    reset_armature(arm)

    prepare_scene(arm)

    # --------------------------------------------------------
    # Restore original transform exactly
    # --------------------------------------------------------

    arm.location = original_location
    arm.rotation_euler = original_rotation
    arm.scale = original_scale

    # --------------------------------------------------------
    # Character scale
    # --------------------------------------------------------

    scale = calculate_scale(
        arm,
        motion
    )

    # --------------------------------------------------------
    # FIRST FRAME = T POSE
    #
    # We use the first recorded frame as calibration.
    # --------------------------------------------------------

    reference = motion[0]

    # --------------------------------------------------------
    # Rest positions
    # --------------------------------------------------------

    rest_right_hand = world_bone_tail(
        arm,
        BONES["right_hand"]
    )

    rest_left_hand = world_bone_tail(
        arm,
        BONES["left_hand"]
    )

    rest_right_foot = world_bone_tail(
        arm,
        BONES["right_foot"]
    )

    rest_left_foot = world_bone_tail(
        arm,
        BONES["left_foot"]
    )

    # --------------------------------------------------------
    # MediaPipe reference positions
    # --------------------------------------------------------

    ref_right_hand = reference[RIGHT_WRIST]
    ref_left_hand = reference[LEFT_WRIST]

    ref_right_foot = reference[RIGHT_ANKLE]
    ref_left_foot = reference[LEFT_ANKLE]

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = create_targets()

    right_hand_target = targets["right_hand"]
    left_hand_target = targets["left_hand"]

    right_foot_target = targets["right_foot"]
    left_foot_target = targets["left_foot"]

    # --------------------------------------------------------
    # Start targets exactly at the character's rest position.
    # --------------------------------------------------------

    set_target(
        right_hand_target,
        rest_right_hand
    )

    set_target(
        left_hand_target,
        rest_left_hand
    )

    set_target(
        right_foot_target,
        rest_right_foot
    )

    set_target(
        left_foot_target,
        rest_left_foot
    )

    # --------------------------------------------------------
    # IK
    # --------------------------------------------------------

    add_ik_target(
        arm,
        BONES["right_hand"],
        right_hand_target,
        3
    )

    add_ik_target(
        arm,
        BONES["left_hand"],
        left_hand_target,
        3
    )

    add_ik_target(
        arm,
        BONES["right_foot"],
        right_foot_target,
        2
    )

    add_ik_target(
        arm,
        BONES["left_foot"],
        left_foot_target,
        2
    )

    # --------------------------------------------------------
    # Keep head and neck stable.
    #
    # This specifically prevents the unwanted head bending
    # you were seeing.
    # --------------------------------------------------------

    head = get_bone(
        arm,
        BONES["head"]
    )

    neck = get_bone(
        arm,
        BONES["neck"]
    )

    head.rotation_mode = 'QUATERNION'
    neck.rotation_mode = 'QUATERNION'

    # --------------------------------------------------------
    # Animate
    # --------------------------------------------------------

    scene = bpy.context.scene

    scene.render.fps = FPS

    scene.frame_start = 1

    scene.frame_end = len(motion)

    print("Creating", len(motion), "animation frames...")

    for i, frame in enumerate(motion):

        blender_frame = i + 1

        scene.frame_set(blender_frame)

        # -----------------------------------------------
        # RIGHT HAND
        # -----------------------------------------------

        right_hand_position = convert_relative(
            frame[RIGHT_WRIST],
            ref_right_hand,
            rest_right_hand,
            scale
        )

        set_target(
            right_hand_target,
            right_hand_position
        )

        # -----------------------------------------------
        # LEFT HAND
        # -----------------------------------------------

        left_hand_position = convert_relative(
            frame[LEFT_WRIST],
            ref_left_hand,
            rest_left_hand,
            scale
        )

        set_target(
            left_hand_target,
            left_hand_position
        )

        # -----------------------------------------------
        # RIGHT FOOT
        # -----------------------------------------------

        right_foot_position = convert_relative(
            frame[RIGHT_ANKLE],
            ref_right_foot,
            rest_right_foot,
            scale
        )

        set_target(
            right_foot_target,
            right_foot_position
        )

        # -----------------------------------------------
        # LEFT FOOT
        # -----------------------------------------------

        left_foot_position = convert_relative(
            frame[LEFT_ANKLE],
            ref_left_foot,
            rest_left_foot,
            scale
        )

        set_target(
            left_foot_target,
            left_foot_position
        )

        # -----------------------------------------------
        # KEYFRAME TARGETS
        # -----------------------------------------------

        right_hand_target.keyframe_insert(
            data_path="location",
            frame=blender_frame
        )

        left_hand_target.keyframe_insert(
            data_path="location",
            frame=blender_frame
        )

        right_foot_target.keyframe_insert(
            data_path="location",
            frame=blender_frame
        )

        left_foot_target.keyframe_insert(
            data_path="location",
            frame=blender_frame
        )

        # -----------------------------------------------
        # Keep head straight
        # -----------------------------------------------

        head.rotation_quaternion = (1, 0, 0, 0)
        neck.rotation_quaternion = (1, 0, 0, 0)

        head.keyframe_insert(
            data_path="rotation_quaternion",
            frame=blender_frame
        )

        neck.keyframe_insert(
            data_path="rotation_quaternion",
            frame=blender_frame
        )

        if blender_frame % 30 == 0:
            print(
                "Animated frame:",
                blender_frame,
                "/",
                len(motion)
            )

    # --------------------------------------------------------
    # REMOVE TARGETS FROM VIEW
    # --------------------------------------------------------

    for obj in targets.values():

        obj.hide_viewport = True
        obj.hide_render = True
        obj.hide_set(True)

    # --------------------------------------------------------
    # HIDE ARMATURE
    # --------------------------------------------------------

    arm.hide_viewport = True
    arm.hide_render = True
    arm.hide_set(True)

    # --------------------------------------------------------
    # MAKE SURE ALL REMY MESHES ARE VISIBLE
    # --------------------------------------------------------

    for obj in bpy.context.scene.objects:

        if obj.type == 'MESH':

            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_set(False)

    # --------------------------------------------------------
    # FRAME 1
    # --------------------------------------------------------

    scene.frame_set(1)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    bpy.ops.wm.save_as_mainfile(
        filepath=str(OUTPUT_FILE)
    )

    print("")
    print("======================================")
    print(" ANIMATION COMPLETE")
    print("======================================")
    print("Frames:", len(motion))
    print("FPS:", FPS)
    print("Saved:")
    print(OUTPUT_FILE)
    print("")
    print("Press SPACEBAR to play animation.")
    print("======================================")


# ============================================================
# RUN
# ============================================================

try:
    animate()

except Exception as e:

    print("")
    print("======================================")
    print("ANIMATION ERROR")
    print("======================================")
    print(str(e))
    print("======================================")

    raise
