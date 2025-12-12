import numpy as np

def elbow_line_angle(left_elbow, right_elbow):
    """
    Returns the acute angle (0–90°) between the elbow–elbow line and horizontal.
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


def sideways_flare_angle(shoulder, elbow, torso_vec):
    """
    Computes sideways flare angle:
    angle between upper arm (shoulder→elbow)
    and outward torso normal (perpendicular to shoulder line).
    """

    # Normalize torso vector (shoulder line)
    torso_vec = torso_vec / (np.linalg.norm(torso_vec) + 1e-8)

    # Outward normal to torso (perpendicular)
    normal_vec = np.array([torso_vec[1], -torso_vec[0]])
    normal_vec = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)

    # Upper arm vector
    upper = elbow - shoulder
    upper = upper / (np.linalg.norm(upper) + 1e-8)

    # Angle between upper arm and torso-normal
    angle = compute_angle(upper, normal_vec)

    # Fold into biomechanical range 0–90°
    if angle > 90:
        angle = 180 - angle

    return float(angle)


def classify_flare(kpts, threshold_deg=70.0):
    """
    - Flared if angle >= threshold_deg.
    - Overall flare = True if either arm crosses threshold.
    """

    # Shoulder keypoints
    L_sh, R_sh = kpts[5], kpts[6]

    # Elbows
    L_el, R_el = kpts[7], kpts[8]

    # Torso line (left shoulder → right shoulder)
    torso_vec = R_sh - L_sh

    # Compute sideways flare angles
    left_angle  = sideways_flare_angle(L_sh, L_el, torso_vec)
    right_angle = sideways_flare_angle(R_sh, R_el, torso_vec)

    # Apply threshold
    left_flared  = bool(left_angle  >= threshold_deg)
    right_flared = bool(right_angle >= threshold_deg)

    # Final classification: either side → flared
    either_flared = bool(left_flared or right_flared)

    return {
        "left_angle_deg":   left_angle,
        "right_angle_deg":  right_angle,
        "threshold_deg":    threshold_deg,
        "left_flared":      left_flared,
        "right_flared":     right_flared,
        "either_flared":    either_flared
    }
