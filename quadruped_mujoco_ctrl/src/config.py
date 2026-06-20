import numpy as np


# ============================================================
# Leg definition
# ============================================================

LEGS = ["FR", "FL", "RR", "RL"]

LEG_INDEX = {
    "FR": 0,
    "FL": 1,
    "RR": 2,
    "RL": 3,
}

LEG_SIDE_SIGN = {
    "FR": -1.0,
    "FL":  1.0,
    "RR": -1.0,
    "RL":  1.0,
}

SUPPORT_LEGS = {
    "FR": ["FL", "RR", "RL"],
    "FL": ["FR", "RR", "RL"],
    "RR": ["FR", "FL", "RL"],
    "RL": ["FR", "FL", "RR"],
}


# ============================================================
# MuJoCo names
# ============================================================

JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

ACTUATOR_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

TOUCH_SENSOR_NAMES = {
    "FR": "fr_touch",
    "FL": "fl_touch",
    "RR": "rr_touch",
    "RL": "rl_touch",
}

FOOT_SITE_NAMES = {
    "FR": "fr_touch_site",
    "FL": "fl_touch_site",
    "RR": "rr_touch_site",
    "RL": "rl_touch_site",
}

HIP_BODY_NAMES = {
    "FR": "FR_hip",
    "FL": "FL_hip",
    "RR": "RR_hip",
    "RL": "RL_hip",
}


# ============================================================
# Model constants
# ============================================================

DT = 0.002

MASS = 12.453
GRAVITY = 9.81
TOTAL_WEIGHT = MASS * GRAVITY


# ============================================================
# Baseline measured from torque standing live probe
# ============================================================

BASELINE_COM_OFFSET_FROM_TRUNK = np.array([
    -0.010889,
     0.001519,
], dtype=float)

HIP_XYZ_BODY = {
    "FR": np.array([ 0.183, -0.047, 0.0], dtype=float),
    "FL": np.array([ 0.183,  0.047, 0.0], dtype=float),
    "RR": np.array([-0.183, -0.047, 0.0], dtype=float),
    "RL": np.array([-0.183,  0.047, 0.0], dtype=float),
}

NOMINAL_FOOT_XYZ_BODY = {
    "FR": np.array([ 0.191088, -0.134604, -0.236855], dtype=float),
    "FL": np.array([ 0.191225,  0.134031, -0.238224], dtype=float),
    "RR": np.array([-0.174912, -0.134317, -0.237941], dtype=float),
    "RL": np.array([-0.175501,  0.133213, -0.239293], dtype=float),
}

NOMINAL_FOOT_XY_BODY = {
    leg: pos[:2].copy()
    for leg, pos in NOMINAL_FOOT_XYZ_BODY.items()
}

BASELINE_FORCE = {
    "FR": 27.13937085330967,
    "FL": 27.843468505921532,
    "RR": 33.55764522930797,
    "RL": 33.62343137432699,
}


# ============================================================
# Body shift limits
# ============================================================

BODY_SHIFT_LIMIT_X = 0.035
BODY_SHIFT_LIMIT_Y = 0.035

BODY_SHIFT_DEFAULT_MAG = 0.04
BODY_SHIFT_MAX_STEP = 0.0015


# ============================================================
# Unload success thresholds
# ============================================================

UNLOAD_FORCE_ABS_THRESHOLD = 10.0
UNLOAD_FORCE_RATIO_THRESHOLD = 0.35

MIN_SUPPORT_FORCE = 20.0
MAX_SUPPORT_FORCE = 65.0

UNLOAD_FEEDBACK_GAIN = 0.015
UNLOAD_FEEDBACK_MAX_CORRECTION = 0.015
UNLOAD_SHIFT_EPS = 1e-6


# ============================================================
# Torque control
# ============================================================

KP_STAND = 100.0
KD_STAND = 3.0
TAU_LIMIT = 33.5


# ============================================================
# QP settings
# ============================================================

FZ_MAX = 70.0
SUPPORT_FZ_MIN = 8.0

SWING_FZ_MIN = 0.0
SWING_FZ_MAX = 8.0

SWING_FORCE_TARGET = 0.0
SWING_FORCE_SOFT_MAX = 8.0

COM_TARGET_LIMIT_X = 0.035
COM_TARGET_LIMIT_Y = 0.035

QP_WEIGHT_FORCE_TRACKING = 0.1
QP_WEIGHT_FORCE_REGULARIZATION = 0.01
QP_WEIGHT_SWING_UNLOAD = 50.0
QP_WEIGHT_COM_SHIFT = 5.0
QP_WEIGHT_SLACK_SUM_FORCE = 1000.0
QP_WEIGHT_SLACK_MOMENT = 1000.0


# ============================================================
# Helper functions
# ============================================================

def get_support_legs(swing_leg: str) -> list[str]:
    if swing_leg not in SUPPORT_LEGS:
        raise ValueError(f"未知 swing_leg: {swing_leg}")
    return SUPPORT_LEGS[swing_leg]


def get_baseline_force_vector() -> np.ndarray:
    return np.array([BASELINE_FORCE[leg] for leg in LEGS], dtype=float)


def get_nominal_foot_xy_matrix() -> np.ndarray:
    return np.vstack([NOMINAL_FOOT_XY_BODY[leg] for leg in LEGS])


def clip_body_shift(body_shift: np.ndarray) -> np.ndarray:
    body_shift = np.asarray(body_shift, dtype=float).copy()

    body_shift[0] = np.clip(
        body_shift[0],
        -BODY_SHIFT_LIMIT_X,
        BODY_SHIFT_LIMIT_X,
    )

    body_shift[1] = np.clip(
        body_shift[1],
        -BODY_SHIFT_LIMIT_Y,
        BODY_SHIFT_LIMIT_Y,
    )

    return body_shift