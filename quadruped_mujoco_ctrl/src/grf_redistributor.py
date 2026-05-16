import numpy as np

LEGS = ["FR", "FL", "RR", "RL"]

FOOT_XY_BODY = {
    "FR": np.array([+0.183, -0.132]),
    "FL": np.array([+0.183, +0.132]),
    "RR": np.array([-0.183, -0.132]),
    "RL": np.array([-0.183, +0.132]),
}
