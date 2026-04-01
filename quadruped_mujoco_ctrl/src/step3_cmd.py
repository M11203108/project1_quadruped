import mujoco
import numpy as np
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from mujoco import viewer
from kinematics import backward_kinematics_2d, forward_kinematics_2d
from cmd_vel_sub import CmdVelSubscriber
from pathlib import Path


x_home, z_home = 0.0, -0.24864398730826576
hip_angle, knee_angle = 0.9, -1.8
hu, hl = 0.2, 0.2
lift_height = 0.035
k_v = 1.0
T = 0.50  # 每 1 秒切換一次
BASE_DIR = Path(__file__).resolve().parents[2]
xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene.xml"
# Load model
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

def get_phase(t, T):
    phase = (t % T) / T # % 取餘數, t=0.2 → t%T = 0.2
    if phase < 0.5:
        active_pair = "A"
        s = phase / 0.5
    else:
        active_pair = "B"
        s = (phase - 0.5) / 0.5
    return phase, active_pair, s

def swing_traj(s, x_home, z_home, step_length, lift_height):
    """
    swing 腳軌跡：
    x: 從後方走到前方
    z: 中間抬高
    """
    x_start = x_home + step_length / 2
    x_end = x_home - step_length / 2

    x = x_start + (x_end - x_start) * s
    z =z_home + lift_height * 4 * s * (1 - s)
    return x, z

def stance_traj(s, x_home, z_home, step_length):
    """
    stance 腳軌跡：
    相對 body 往後掃，模擬身體往前走
    z: 保持不動
    """
    x_start = x_home + step_length / 2
    x_end = x_home - step_length / 2
    # x = x_start + (x_end - x_start) * s
    x = x_home
    z = z_home
    return x, z

def set_leg_ctrl(ctrl, hip_id, knee_id, x, z, hu, hl, ctrl_range):
    hip_angle, knee_angle = backward_kinematics_2d(x, z, hu, hl)
    hip_angle = np.clip(hip_angle, ctrl_range[hip_id, 0], ctrl_range[hip_id, 1])
    knee_angle = np.clip(knee_angle, ctrl_range[knee_id, 0], ctrl_range[knee_id, 1])
    ctrl[hip_id] = hip_angle
    ctrl[knee_id] = knee_angle

def main():
    rclpy.init()
    cmd_node = CmdVelSubscriber()
    # print(backward_kinematics_2d(x_home, z_home, hu, hl))
    # print(np.rad2deg(backward_kinematics_2d(x_home, z_home, hu, hl)))

    x, z = forward_kinematics_2d(hip_angle, knee_angle, hu, hl)
    print("FK:", x, z)

    hip2, knee2 = backward_kinematics_2d(x, z, hu, hl)
    print("IK:", hip2, knee2)

    print("FK->IK degree:", np.rad2deg([hip2, knee2]))

    # Load the MuJoCo model from an XML file

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name) # 取得 keyframe 的 ID
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data) # 計算正向動力學，更新 data.qpos 和 data.qvel
    ctrl_home = data.ctrl.copy() # 取得當前的控制輸入，作為目標控制輸入
    ctrl_range = model.actuator_ctrlrange.copy() # 取得控制輸入的範圍

    # 找 FR 的 actuator index
    fr_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_thigh")
    fr_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_calf")
    fl_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_thigh")
    fl_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_calf")
    rr_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_thigh")
    rr_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_calf")
    rl_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_thigh")
    rl_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_calf")

    t0 = time.perf_counter() # 記錄起始時間

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            rclpy.spin_once(cmd_node, timeout_sec=0.0) # 讓 ROS2 處理一次 callback，更新 cmd_node 的 cmd_vel
            cmd_linear_x = cmd_node.cmd_linear_x # 前進為正，後退為負
            cmd_angular_z = cmd_node.cmd_angular_z # 逆時針轉為正，順時針轉為負
            linear_deadzone = 0.05
            angular_deadzone = 0.1
            if abs(cmd_linear_x) < linear_deadzone and abs(cmd_angular_z) < angular_deadzone:
                data.ctrl[:] = ctrl_home
                mujoco.mj_step(model, data)
                v.sync()
                continue
            foward_step = k_v * cmd_linear_x * T
            turn_step = k_v * cmd_angular_z * T
            left_step_length = foward_step - turn_step
            right_step_length = foward_step + turn_step
            t =time.perf_counter() - t0 # 計算經過的時間
            phase, active_pair, s = get_phase(t, T)

            ctrl = ctrl_home.copy()
        

            if active_pair == "A":
                x_fr, z_fr = swing_traj(s, x_home, z_home, right_step_length, lift_height)
                x_rl, z_rl = swing_traj(s, x_home, z_home, left_step_length, lift_height)

                x_rr, z_rr = stance_traj(s, x_home, z_home, right_step_length)
                x_fl, z_fl = stance_traj(s, x_home, z_home, left_step_length)

            else:
                x_fl, z_fl = swing_traj(s, x_home, z_home, left_step_length, lift_height)
                x_rr, z_rr = swing_traj(s, x_home, z_home, right_step_length, lift_height)

                x_fr, z_fr = stance_traj(s, x_home, z_home, right_step_length)
                x_rl, z_rl = stance_traj(s, x_home, z_home, left_step_length)

            set_leg_ctrl(ctrl, fr_hip, fr_knee, x_fr, z_fr, hu, hl, ctrl_range)
            set_leg_ctrl(ctrl, fl_hip, fl_knee, x_fl, z_fl, hu, hl, ctrl_range)
            set_leg_ctrl(ctrl, rr_hip, rr_knee, x_rr, z_rr, hu, hl, ctrl_range)
            set_leg_ctrl(ctrl, rl_hip, rl_knee, x_rl, z_rl, hu, hl, ctrl_range)

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            v.sync()
        cmd_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()