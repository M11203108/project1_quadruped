import numpy as np

import config as cfg
from foot_target_builder import FootTargetBuilder

builder = FootTargetBuilder()

body_shift_cmd = np.array([0.05, -0.05])
swing_leg = "RL"

targets = builder.build(
    body_shift_cmd=body_shift_cmd,
    swing_leg=swing_leg,
    lift_height=0.0,
    apply_shift_to_swing=False,
)

print("body_shift_cmd:", body_shift_cmd)
print("swing_leg:", swing_leg)
print("apply_shift_to_swing: False")

for leg in cfg.LEGS:
    nominal = cfg.NOMINAL_FOOT_XYZ_BODY[leg]
    target = targets[leg]
    delta = target - nominal

    print(
        leg,
        "nominal:", np.round(nominal, 4),
        "target:", np.round(target, 4),
        "delta:", np.round(delta, 4),
    )

print("\nNow test lift height")

targets_lift = builder.build(
    body_shift_cmd=body_shift_cmd,
    swing_leg=swing_leg,
    lift_height=0.03,
    apply_shift_to_swing=False,
)

for leg in cfg.LEGS:
    nominal = cfg.NOMINAL_FOOT_XYZ_BODY[leg]
    target = targets_lift[leg]
    delta = target - nominal

    print(
        leg,
        "target:", np.round(target, 4),
        "delta:", np.round(delta, 4),
    )