import numpy as np

def elbow_line_angle(left_elbow, right_elbow):
    """
    Returns the acute angle (0–90°) between the elbow–elbow line and horizontal.
    This is the correct metric for bench press symmetry.
    """
    dx = right_elbow[0] - left_elbow[0]
    dy = right_elbow[1] - left_elbow[1]

    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))  # acute angle
    return float(angle)

def analyze_elbow_line(kpts, level_threshold_deg=10.0):
    """
    Computes:
      - angle of elbow-elbow line relative to horizontal
      - absolute deviation
      - classification (level / not level)
    """

    left_elbow  = kpts[7]
    right_elbow = kpts[8]

    angle = elbow_line_angle(left_elbow, right_elbow)
    abs_angle = abs(angle)

    is_level = abs_angle <= level_threshold_deg

    return {
        "elbow_line_angle_deg": angle,
        "elbow_line_angle_abs_deg": abs_angle,
        "is_level": is_level,
        "level_threshold_deg": level_threshold_deg
    }

def compute_angle(v1, v2):
    dot = float(np.dot(v1, v2))
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cos_val = np.clip(dot / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def barbell_normal(kpts):
    L_wr, R_wr = kpts[9], kpts[10]
    bar_vec = R_wr - L_wr
    bar_vec = bar_vec / (np.linalg.norm(bar_vec) + 1e-8)

    # outward lateral direction
    normal = np.array([bar_vec[1], -bar_vec[0]])
    return normal / (np.linalg.norm(normal) + 1e-8)


def sideways_flare_angle(shoulder, elbow, normal):
    upper = elbow - shoulder
    upper = upper / (np.linalg.norm(upper) + 1e-8)

    angle = compute_angle(upper, normal)
    if angle > 90:
        angle = 180 - angle
    return angle


def elbow_drop_metric(shoulder, elbow, shoulder_width):
    # elbow dropping sideways/backwards relative to shoulder level
    vertical = elbow[1] - shoulder[1]
    return vertical / (shoulder_width + 1e-8)


def classify_flare(kpts, side_threshold=75.0, drop_threshold=0.45):
    """
    NEW BEHAVIOR:

    - Flare is determined ONLY by sideways angle.
    - Drop no longer triggers flare.
    - Drop triggers a photo-quality warning suggesting a retake.

    Returns both:
      - flare flags
      - retake suggestion flags
    """
    L_sh, R_sh = kpts[5], kpts[6]
    L_el, R_el = kpts[7], kpts[8]

    shoulder_width = float(np.linalg.norm(R_sh - L_sh))
    normal = barbell_normal(kpts)

    # SIDEWAYS flare angle (primary biomechanical signal)
    L_side = float(sideways_flare_angle(L_sh, L_el, normal))
    R_side = float(sideways_flare_angle(R_sh, R_el, normal))

    # DROP metric (camera/lighting/angle guardrail)
    L_drop = float(elbow_drop_metric(L_sh, L_el, shoulder_width))
    R_drop = float(elbow_drop_metric(R_sh, R_el, shoulder_width))

    # Flare is ONLY sideways-based now
    L_flared = bool(L_side > side_threshold)
    R_flared = bool(R_side > side_threshold)

    # Retake suggestion is ONLY drop-based now
    L_retake = bool(L_drop > drop_threshold)
    R_retake = bool(R_drop > drop_threshold)

    return {
        "left_side_angle_deg":  L_side,
        "right_side_angle_deg": R_side,
        "left_drop_norm":       L_drop,
        "right_drop_norm":      R_drop,

        "left_flared":          L_flared,
        "right_flared":         R_flared,

        # NEW fields for UX
        "left_retake_suggested":  L_retake,
        "right_retake_suggested": R_retake,
        "retake_suggested":       bool(L_retake or R_retake),

        "side_threshold_deg":   float(side_threshold),
        "drop_threshold_norm":  float(drop_threshold),

        # Optional: a human-readable hint for the app
        "note": (
            "Flare is determined by sideways angle only. "
            "Drop is used as a camera/angle/lighting guardrail."
        )
    }

# Old elbow flare implementation
"""
def compute_angle(v1, v2):
    
    Returns angle between two vectors in degrees.
    Always in 0–180°, clipping used to avoid NaNs.
    
    dot = float(np.dot(v1, v2))
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cos_val = np.clip(dot / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def elbow_flare_angle(shoulder, elbow, torso_vec):
    Computes elbow flare angle using the CORRECT biomechanical definition:

    flare angle = angle between:
        - the upper-arm vector (shoulder → elbow)
        - the outward normal of the shoulder-to-shoulder line

    This works for front-view, incline bench, and most gym camera angles.

    # Normalize torso (shoulder line)
    torso_vec = torso_vec / (np.linalg.norm(torso_vec) + 1e-8)

    # Perpendicular (normal) vector to torso line: OUTWARD direction
    normal = np.array([torso_vec[1], -torso_vec[0]])
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Upper arm vector
    upper = elbow - shoulder
    upper = upper / (np.linalg.norm(upper) + 1e-8)

    # Compute flare angle (0–180°)
    angle = compute_angle(upper, normal)

    # Fold into 0–90° biomechanical range
    if angle > 90:
        angle = 180 - angle

    return float(angle)


def classify_flare(kpts, threshold=40.0):
    
    Classifies elbow flare given YOLO keypoints for a single person.

    Assumes image is from the bottom of the bench press (lowered),
    so flare evaluation is always performed.
    

    # Shoulders
    L_sh = kpts[5]
    R_sh = kpts[6]

    # Elbows
    L_el = kpts[7]
    R_el = kpts[8]

    # Shoulder line: left → right
    torso_line = R_sh - L_sh

    # Compute angles
    left_angle  = elbow_flare_angle(L_sh, L_el, torso_line)
    right_angle = elbow_flare_angle(R_sh, R_el, torso_line)

    return {
        "left_angle_deg":  left_angle,
        "right_angle_deg": right_angle,
        "left_flared":     left_angle  > threshold,
        "right_flared":    right_angle > threshold,
        "threshold_deg":   threshold
    }
"""