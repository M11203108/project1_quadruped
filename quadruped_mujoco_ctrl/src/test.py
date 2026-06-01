import mujoco
import numpy as np
from mujoco import viewer
from pathlib import Path

from robot_interface import RobotInterface, LEGS
from ik_controller import IKController

def reset_to_home(robot):
    key_id = mujoco.mj_name2id(
        robot.model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if key_id < 0:
        raise RuntimeError("找不到 keyframe: home")

    mujoco.mj_resetDataKeyframe(robot.model, robot.data, key_id)
    robot.forward()

    print("已 reset 到 home keyframe")


def compute_pd_hold_torque(robot, state, q_des, Kp=60.0, Kd=2.0):
    qd_des = np.zeros(12)

    tau_pd = Kp * (q_des - state.q) + Kd * (qd_des - state.qd)

    # qfrc_bias 是 MuJoCo 算出來的 bias / gravity compensation
    tau_ff = np.array(
        [
            robot.data.qfrc_bias[i]
            for i in robot.ids["qvel"]
        ]
    )

    tau = tau_ff + tau_pd

    return tau


def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    xml_path = (
        BASE_DIR
        / "third_party"
        / "mujoco_menagerie"
        / "unitree_a1"
        / "scene_torque.xml"
    )

    robot = RobotInterface(xml_path)

    reset_to_home(robot)

    # reset 到 home 後，讀目前 12 維關節角，當作站立目標
    state = robot.read_state()
    q_des_home = state.q.copy()

    step_count = 0

    ik = IKController(
    h=0.08505,
    hu=0.2,
    hl=0.2,
    )

    with viewer.launch_passive(robot.model, robot.data) as v:
        while v.is_running():
            state = robot.read_state()

            # 1. PD 繼續追 home 姿態
            tau = compute_pd_hold_torque(
                robot=robot,
                state=state,
                q_des=q_des_home,
                Kp=60.0,
                Kd=2.0,
            )

            tau_safe = robot.write_torque(tau)

            # 2. IK 只做測試，不拿來控制
            foot_targets_body = state.foot_xyz_body.copy()

            q_des_ik = ik.solve(
                foot_targets_body=foot_targets_body,
                hip_xyz_body=state.hip_xyz_body,
            )

            ik_err = np.max(np.abs(q_des_ik - state.q))
            home_err = np.max(np.abs(q_des_home - state.q))

            robot.step()
            v.sync()

            if step_count % 50 == 0:
                state = robot.read_state()

                forces_str = {
                    leg: round(state.forces[leg], 2)
                    for leg in LEGS
                }

                print(
                    "step:", step_count,
                    "trunk_z:", round(state.trunk_pos_world[2], 3),
                    "forces:", forces_str,
                    "max_tau:", round(np.max(np.abs(tau_safe)), 2),
                    "home_err:", round(home_err, 3),
                    "ik_err:", round(ik_err, 3),
                    "q_des_ik_shape:", q_des_ik.shape,
                )

            step_count += 1


if __name__ == "__main__":
    main()