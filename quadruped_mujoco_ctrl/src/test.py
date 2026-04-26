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
from sensor_msgs.msg import Imu


x_home, z_home = 0.0, -0.24864398730826576
hip_angle, knee_angle = 0.9, -1.8
hu, hl = 0.2, 0.2
lift_height = 0.06
k_lin = 1.0
k_yaw = 0.25
T = 0.8  # 每 1 秒切換一次
BASE_DIR = Path(__file__).resolve().parents[2]
xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene.xml"
# Load model
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
# model.opt.gravity[:] = [0.0, 0.0, 0.0]
print("nsensor =", model.nsensor)

LEG_CFG = {
    "FR": {"group": "front", "hip_sign": 1.0, "knee_sign": 1.0, "hip_offset": 0.0, "knee_offset": 0.0},
    "FL": {"group": "front", "hip_sign": 1.0, "knee_sign": 1.0, "hip_offset": 0.0, "knee_offset": 0.0},
    "RR": {"group": "rear",  "hip_sign": 1.0, "knee_sign": 1.0, "hip_offset": 0.0, "knee_offset": 0.0},
    "RL": {"group": "rear",  "hip_sign": 1.0, "knee_sign": 1.0, "hip_offset": 0.0, "knee_offset": 0.0},
}

def get_phase(t, T):
    phase = (t % T) / T # % 取餘數, t=0.2 → t%T = 0.2
    if phase < 0.5:
        active_pair = "A"
        s = phase / 0.5
    else:
        active_pair = "B"
        s = (phase - 0.5) / 0.5
    return phase, active_pair, s

def ik_front_2d(x_body, z_body, hu, hl):
    x_leg = -x_body
    z_leg = z_body
    hip, knee = backward_kinematics_2d(x_leg, z_leg, hu, hl)
    return hip, knee

def ik_rear_2d(x_body, z_body, hu, hl):
    x_leg = -x_body
    z_leg = z_body
    hip, knee = backward_kinematics_2d(x_leg, z_leg, hu, hl)
    return hip, knee

def ik_front_cmd(x_body, z_body, hu, hl):
    hip, knee = ik_front_2d(x_body, z_body, hu, hl)
    hip_cmd = hip
    knee_cmd = knee
    return hip_cmd, knee_cmd

def ik_rear_cmd(x_body, z_body, hu, hl):
    hip, knee = ik_rear_2d(x_body, z_body, hu, hl)
    hip_cmd = hip
    knee_cmd = knee
    return hip_cmd, knee_cmd

def swing_traj(s, x_home, z_home, step_length, lift_height):
    """
    swing 腳軌跡：
    x: 從後方走到前方
    z: 中間抬高
    """
    x_start = x_home - step_length / 2
    x_end = x_home + step_length / 2

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
    x = x_start + (x_end - x_start) * s
    # x = x_home
    z = z_home
    return x, z

def solve_leg_ik(leg_name, x_body, z_body, hu, hl):
    cfg = LEG_CFG[leg_name]

    if cfg["group"] == "front":
        hip, knee = ik_front_2d(x_body, z_body, hu, hl)
    else:
        hip, knee = ik_rear_2d(x_body, z_body, hu, hl)

    hip = cfg["hip_sign"] * hip + cfg["hip_offset"]
    knee = cfg["knee_sign"] * knee + cfg["knee_offset"]
    return hip, knee

def set_leg_ctrl(ctrl, leg_name, hip_id, knee_id, x_body, z_body, hu, hl, ctrl_range):
    hip_raw, knee_raw = solve_leg_ik(leg_name, x_body, z_body, hu, hl)

    hip_cmd = np.clip(hip_raw, ctrl_range[hip_id, 0], ctrl_range[hip_id, 1])
    knee_cmd = np.clip(knee_raw, ctrl_range[knee_id, 0], ctrl_range[knee_id, 1])

    if abs(hip_cmd - hip_raw) > 1e-6 or abs(knee_cmd - knee_raw) > 1e-6:
        print(f"[CLIP] {leg_name} x={x_body:.4f}, z={z_body:.4f}, "
              f"hip_raw={hip_raw:.4f}, hip_clip={hip_cmd:.4f}, "
              f"knee_raw={knee_raw:.4f}, knee_clip={knee_cmd:.4f}")

    ctrl[hip_id] = hip_cmd
    ctrl[knee_id] = knee_cmd

def main():
    rclpy.init()
    cmd_node = CmdVelSubscriber()
    # print(backward_kinematics_2d(x_home, z_home, hu, hl))
    # print(np.rad2deg(backward_kinematics_2d(x_home, z_home, hu, hl)))


    x, z = forward_kinematics_2d(hip_angle, knee_angle, hu, hl)
    # print("FK:", x, z)

    hip2, knee2 = backward_kinematics_2d(x, z, hu, hl)
    # print("IK:", hip2, knee2)

    # print("FK->IK degree:", np.rad2deg([hip2, knee2]))

    # Load the MuJoCo model from an XML file

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name) # 取得 keyframe 的 ID
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    mujoco.mj_forward(model, data) # 計算正向動力學，更新 data.qpos 和 data.qvel

    ctrl_home = data.ctrl.copy() # 取得當前的控制輸入，作為目標控制輸入
    ctrl_range = model.actuator_ctrlrange.copy() # 取得控制輸入的範圍

    fr_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_thigh")
    fr_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_calf")
    fl_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_thigh")
    fl_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_calf")
    rr_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_thigh")
    rr_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_calf")
    rl_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_thigh")
    rl_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_calf")
    #側向
    fr_abd = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_hip")
    fl_abd = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_hip")
    rr_abd = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_hip")
    rl_abd = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_hip")


    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            rclpy.spin_once(cmd_node, timeout_sec=0.0) # 讓 ROS2 處理一次 callback，更新 cmd_node 的 cmd_vel
            cmd_linear_x = cmd_node.cmd_linear_x # 前進為正，後退為負
            cmd_angular_z = cmd_node.cmd_angular_z # 逆時針轉為正，順時針轉為負
            linear_deadzone = 0.05
            angular_deadzone = 0.1
            # if abs(cmd_linear_x) < linear_deadzone and abs(cmd_angular_z) < angular_deadzone:
            #     data.ctrl[:] = ctrl_home
            #     mujoco.mj_step(model, data)
            #     v.sync()
            #     continue
            
            max_step = 0.03
            max_turn = 0.02

            # base_step = np.clip(k_lin * cmd_linear_x, -max_step, max_step)
            # turn_step = np.clip(k_yaw * cmd_angular_z, -max_turn, max_turn)
            # left_step_length = base_step - turn_step
            # right_step_length = base_step + turn_step
            # t = data.time
            # phase, active_pair, s = get_phase(t, T)
            # # print("cmd_linear_x:", cmd_linear_x, "cmd_angular_z:", cmd_angular_z, "left_step_length:", left_step_length, "right_step_length:", right_step_length)

            # ctrl = ctrl_home.copy()


            # if active_pair == "A":
            #     x_fr, z_fr = swing_traj(s, x_home, z_home, right_step_length, lift_height)
            #     x_rl, z_rl = swing_traj(s, x_home, z_home, left_step_length, lift_height)

            #     x_rr, z_rr = stance_traj(s, x_home, z_home, right_step_length)
            #     x_fl, z_fl = stance_traj(s, x_home, z_home, left_step_length)

            # else:
            #     x_fl, z_fl = swing_traj(s, x_home, z_home, left_step_length, lift_height)
            #     x_rr, z_rr = swing_traj(s, x_home, z_home, right_step_length, lift_height)

            #     x_fr, z_fr = stance_traj(s, x_home, z_home, right_step_length)
            #     x_rl, z_rl = stance_traj(s, x_home, z_home, left_step_length)

            # set_leg_ctrl(ctrl, "FR", fr_thigh, fr_calf, x_fr, z_fr, hu, hl, ctrl_range)
            # set_leg_ctrl(ctrl, "FL", fl_thigh, fl_calf, x_fl, z_fl, hu, hl, ctrl_range)
            # set_leg_ctrl(ctrl, "RR", rr_thigh, rr_calf, x_rr, z_rr, hu, hl, ctrl_range)
            # set_leg_ctrl(ctrl, "RL", rl_thigh, rl_calf, x_rl, z_rl, hu, hl, ctrl_range)
            ctrl = ctrl_home.copy()

            cycle_T = 1.2
            u = (data.time % cycle_T) / cycle_T   # 0~1

            x_unload = 0.015   # 支撐腳往前推
            z_press  = 0.008   # 支撐腳往下壓
            z_lift   = 0.020   # FR 抬腳高度

            # 先全部回到 home
            x_fr = x_home
            x_fl = x_home
            x_rr = x_home
            x_rl = x_home

            z_fr = z_home
            z_fl = z_home
            z_rr = z_home
            z_rl = z_home

            # Phase 1: 卸重（0 ~ 0.35）
            if u < 0.35:
                a = u / 0.35
                support_x = x_home + x_unload * a
                support_z = z_home - z_press * a

                x_fl, z_fl = support_x, support_z
                x_rr, z_rr = support_x, support_z
                x_rl, z_rl = support_x, support_z

            # Phase 2: 保持卸重，同時抬 FR（0.35 ~ 0.75）
            elif u < 0.75:
                a = (u - 0.35) / 0.40   # 0~1

                x_fl, z_fl = x_home + x_unload, z_home - z_press
                x_rr, z_rr = x_home + x_unload, z_home - z_press
                x_rl, z_rl = x_home + x_unload, z_home - z_press

                z_fr = z_home + z_lift * 4 * a * (1 - a)

            # Phase 3: 支撐腳回來（0.75 ~ 1.0）
            else:
                a = (u - 0.75) / 0.25
                support_x = x_home + x_unload * (1.0 - a)
                support_z = z_home - z_press * (1.0 - a)

                x_fl, z_fl = support_x, support_z
                x_rr, z_rr = support_x, support_z
                x_rl, z_rl = support_x, support_z

            set_leg_ctrl(ctrl, "FR", fr_thigh, fr_calf, x_fr, z_fr, hu, hl, ctrl_range)
            set_leg_ctrl(ctrl, "FL", fl_thigh, fl_calf, x_fl, z_fl, hu, hl, ctrl_range)
            set_leg_ctrl(ctrl, "RR", rr_thigh, rr_calf, x_rr, z_rr, hu, hl, ctrl_range)
            set_leg_ctrl(ctrl, "RL", rl_thigh, rl_calf, x_rl, z_rl, hu, hl, ctrl_range)

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            v.sync()
        cmd_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()