import time
import numpy as np
import mujoco

import config as cfg
from pathlib import Path

from robot_state import build_mujoco_ids, read_robot_state
from unload_planner import UnloadPlanner
from body_shift_controller import BodyShiftController
from foot_target_builder import FootTargetBuilder
from joint_pd_controller import JointPDController
from ik_controller import IKController


BASE_DIR = Path(__file__).resolve().parents[2]
XML_PATH = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"
SWING_LEG = "RL"


def reset_model(model, data):
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    reset_model(model, data)

    ids = build_mujoco_ids(model)

    unload_planner = UnloadPlanner(enable_force_feedback=False)
    body_shift_controller = BodyShiftController()
    foot_target_builder = FootTargetBuilder()
    ik_controller = IKController()
    joint_pd_controller = JointPDController()

    for step in range(3000):
        state = read_robot_state(model, data, ids)

        plan = unload_planner.plan(
            state=state,
            swing_leg=SWING_LEG,
        )

        body_shift_cmd = body_shift_controller.update(
            plan.body_shift_target,
        )

        foot_targets_body = foot_target_builder.build(
            body_shift_cmd=body_shift_cmd,
            swing_leg=SWING_LEG,
            lift_height=0.0,
            apply_shift_to_swing=False,
        )

        q_des = ik_controller.solve(
            foot_targets_body=foot_targets_body,
            hip_xyz_body=cfg.HIP_XYZ_BODY,
        )

        pd_result = joint_pd_controller.compute_from_state(
            state=state,
            q_des=q_des,
            data=data,
            ids=ids,
        )

        joint_pd_controller.write_to_mujoco(
            data=data,
            ids=ids,
            tau_cmd=pd_result.tau_cmd,
        )

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            print("================================")
            print("step:", step)
            print("time:", round(float(data.time), 3))
            print("plan reason:", plan.reason)
            print("unload_success:", plan.unload_success)
            print("body_shift_target:", np.round(plan.body_shift_target, 4))
            print("body_shift_cmd:", np.round(body_shift_cmd, 4))
            print("forces:", {
                leg: round(float(state.forces[leg]), 2)
                for leg in cfg.LEGS
            })
            print("swing_force:", round(plan.swing_force, 2))
            print("swing_ratio:", round(plan.swing_force_ratio, 3))
            print("support_ok:", plan.support_ok)
            print("max_tau:", round(pd_result.max_abs_tau_cmd, 3))
            print("saturated:", pd_result.saturated)

        time.sleep(0.001)


if __name__ == "__main__":
    main()