import numpy as np

from body_shift_controller import BodyShiftController

ctrl = BodyShiftController()

target = np.array([0.05, -0.05])

print("target:", target)

for i in range(40):
    cmd = ctrl.update(target)

    print(
        "step:", i,
        "cmd:", np.round(cmd, 4),
        "error:", np.round(ctrl.get_error(), 4),
        "reached:", ctrl.is_target_reached(),
    )