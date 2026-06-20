import numpy as np

import config as cfg
from unload_planner import UnloadPlanner


class FakeState:
    def __init__(self):
        self.com_xy_body = cfg.BASELINE_COM_OFFSET_FROM_TRUNK.copy()
        self.foot_xy_body = cfg.NOMINAL_FOOT_XY_BODY
        self.forces = cfg.BASELINE_FORCE.copy()


planner = UnloadPlanner()

for swing_leg in cfg.LEGS:
    state = FakeState()

    result = planner.plan(
        state=state,
        swing_leg=swing_leg,
    )

    print("\n====================")
    print("swing_leg:", swing_leg)
    print("success:", result.success)
    print("reason:", result.reason)
    print("body_shift_target:", np.round(result.body_shift_target, 5))
    print("unload_success:", result.unload_success)
    print("support_ok:", result.support_ok)
    print("swing_force:", round(result.swing_force, 3))
    print("swing_force_ratio:", round(result.swing_force_ratio, 3))

    print("desired_forces:")
    for leg, f in result.desired_forces.items():
        print(leg, round(f, 3))

    print("force_error:")
    for leg, e in result.force_error.items():
        print(leg, round(e, 3))

    print("feedback_correction:", np.round(result.feedback_correction, 5))