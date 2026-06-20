import numpy as np

import config as cfg
from com_grf_qp_solver import CoMGRFQPSolver

solver = CoMGRFQPSolver()

for swing_leg in cfg.LEGS:
    result = solver.solve(
        swing_leg=swing_leg,
        foot_xy_body=cfg.NOMINAL_FOOT_XY_BODY,
        current_com_xy_body=cfg.BASELINE_COM_OFFSET_FROM_TRUNK,
        measured_forces=cfg.BASELINE_FORCE,
    )

    print("\n====================")
    print("swing_leg:", swing_leg)
    print("success:", result.success)
    print("status:", result.status)

    print("com_target_xy:", np.round(result.com_target_xy, 5))
    print(
        "body_shift_est:",
        np.round(result.com_target_xy - cfg.BASELINE_COM_OFFSET_FROM_TRUNK, 5),
    )

    print("desired_forces:")
    for leg, f in result.desired_forces.items():
        print(leg, round(f, 3))

    print("sum force:", round(float(np.sum(result.desired_force_vector)), 3))
    print("slack:", np.round(result.slack, 6))