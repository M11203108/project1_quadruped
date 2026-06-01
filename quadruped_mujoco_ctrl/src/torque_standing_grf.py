import mujoco
import numpy as np
from mujoco import viewer
from pathlib import Path
from kinematics import backward_kinematics_3d

LEGS = ["FR", "FL", "RR", "RL"]

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

SITE_NAMES = {
    "FR": "fr_touch_site",
    "FL": "fl_touch_site",
    "RR": "rr_touch_site",
    "RL": "rl_touch_site",
}

LEG_SIDE_SIGN = {
    "FR": -1.0,
    "RR": -1.0,
    "FL":  1.0,
    "RL":  1.0,
}

def must_find_id(model, obj_type, name):
    """
    用名字找 ID
    """
    obj_id = mujoco.mj_name2id(model, obj_type, name)

    if obj_id < 0:
        raise RuntimeError(f"找不到 MuJoCo 物件: {name}")

    return obj_id

def get_ids(model):
    qpos_ids = []
    qvel_ids = []
    actuator_ids = []

    for joint_name, actuator_name in zip(JOINT_NAMES, ACTUATOR_NAMES):         

        joint_id = must_find_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = must_find_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        qpos_id = model.jnt_qposadr[joint_id]
        qvel_id = model.jnt_dofadr[joint_id]

        qpos_ids.append(qpos_id)
        qvel_ids.append(qvel_id)
        actuator_ids.append(actuator_id)

    site_ids = {}
    for leg, site_name in SITE_NAMES.items():

        site_id = must_find_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        site_ids[leg] = site_id

    touch_adrs = {}

    for leg, sensor_name in TOUCH_SENSOR_NAMES.items():
        
        sensor_id = must_find_id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        sensor_adr = int(model.sensor_adr[sensor_id])
        touch_adrs[leg] = sensor_adr

    ids = {
        "qpos": qpos_ids,
        "qvel": qvel_ids,
        "actuator": actuator_ids,
        "site": site_ids,
        "touch_adr": touch_adrs,
    }

    return ids

def read_touch_forces(data, ids):
    forces = {}
    for leg in LEGS:
        adr = ids["touch_adr"][leg]
        force = float(data.sensordata[adr])
        forces[leg] = force

    return forces

def get_foot_xy_body(model, data, ids):
    """
    取得四隻腳 foot site 在 body/trunk frame 下的 xy 位置。

    return:
        foot_xy_body = {
            "FR": np.array([x, y]),
            "FL": np.array([x, y]),
            "RR": np.array([x, y]),
            "RL": np.array([x, y]),
        }
    """
    trunk_id = must_find_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "trunk",
    )

    trunk_pos_world = data.xpos[trunk_id].copy()
    trunk_rot_world = data.xmat[trunk_id].reshape(3, 3).copy()

    foot_xy_body = {}

    for leg in LEGS:
        site_id = ids["site"][leg]

        foot_pos_world = data.site_xpos[site_id].copy()

        foot_pos_body = trunk_rot_world.T @ (
            foot_pos_world - trunk_pos_world
        )

        foot_xy_body[leg] = foot_pos_body[:2].copy()

    return foot_xy_body

def get_foot_xyz_body(model, data, ids):
    trunk_id = must_find_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "trunk",
    )

    trunk_pos_world = data.xpos[trunk_id].copy()
    trunk_rot_world = data.xmat[trunk_id].reshape(3, 3).copy()

    foot_xyz_body = {}

    for leg in LEGS:
        site_id = ids["site"][leg]

        foot_pos_world = data.site_xpos[site_id].copy()

        foot_pos_body = trunk_rot_world.T @ (
            foot_pos_world - trunk_pos_world
        )

        foot_xyz_body[leg] = foot_pos_body.copy()

    return foot_xyz_body

def leg_ik_3d(leg, foot_target_body, hip_body, h, hu, hl):
    """
    foot target 從 body frame 轉成 leg/hip frame
    關節角度
    """

    foot_target_body = np.asarray(foot_target_body, dtype=float)
    hip_body = np.asarray(hip_body, dtype=float)
    # body frame → 單腳 hip frame
    foot_target_leg = foot_target_body - hip_body

    side_angle = LEG_SIDE_SIGN[leg]

    abd, hip, knee = backward_kinematics_3d(
        foot_target_leg[0],
        foot_target_leg[1],
        foot_target_leg[2],
        h,
        hu,
        hl,
        side_angle=side_angle,
    )

    return np.array([abd, hip, knee], dtype=float)

def foot_targets_to_q_des(foot_targets_body, hip_xyz_body, h, hu, hl):
    """
    foot target 轉 12 維 q_des
    """

    q_des_list = []

    for leg in LEGS:
        q_leg = leg_ik_3d(
            leg,
            foot_targets_body[leg],
            hip_xyz_body[leg],
            h,
            hu,
            hl,
        )

        q_des_list.extend(q_leg)

    q_des = np.array(q_des_list, dtype=float)

    if q_des.shape != (12,):
        raise RuntimeError(f"q_des shape 錯誤，目前是 {q_des.shape}")

    return q_des

def get_hip_xyz_body(model, data):
    """
    取得四隻腳 hip joint anchor 在 body/trunk frame 下的 xyz 位置
    """

    trunk_id = must_find_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "trunk",
    )

    trunk_pos_world = data.xpos[trunk_id].copy()
    trunk_rot_world = data.xmat[trunk_id].reshape(3, 3).copy()

    hip_xyz_body = {}

    for leg in LEGS:
        hip_joint_name = f"{leg}_hip_joint"

        hip_joint_id = must_find_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            hip_joint_name,
        )

        # MuJoCo 會把 joint anchor 的世界座標存在 data.xanchor
        hip_pos_world = data.xanchor[hip_joint_id].copy()

        # world frame → trunk/body frame
        hip_pos_body = trunk_rot_world.T @ (
            hip_pos_world - trunk_pos_world
        )

        hip_xyz_body[leg] = hip_pos_body.copy()

    return hip_xyz_body

def read_joint_states(data, ids):
    """
    q:12關節角度
    qd:12關節速度
    """
    q = np.array([data.qpos[i] for i in ids["qpos"]])
    qd = np.array([data.qvel[i] for i in ids["qvel"]])
    return q, qd

def compute_standing_torque(data, ids, q, qd, q_des, qd_des, Kp, Kd):
    """
    站立控制
    """
    tau_pd = Kp * (q_des - q) + Kd * (qd_des - qd)
    tau_bias = np.array([data.qfrc_bias[i] for i in ids["qvel"]])
    tau = tau_bias + tau_pd
    return tau

def write_torque(data, ids, tau, tau_limit=33.5):

    tau = np.clip(tau, -tau_limit, tau_limit)
    for i, actuator_id in enumerate(ids["actuator"]):
        data.ctrl[actuator_id] = tau[i]
    return tau

def compute_desired_grf(forces, mode="shift_to_FR"):
    """
    forces:
        measured touch forces
    return:
        desired_forces
    """
    total_force = 0.0
    for leg in LEGS:
        total_force += max(float(forces[leg]), 0.0)
    if total_force < 1e-6:
        total_force = 120.0
    average_force = total_force / 4.0
    desired_forces = {
        leg: average_force
        for leg in LEGS
    }

    if mode == "shift_to_FR":
        delta = 15.0

        desired_forces["FR"] += delta
        desired_forces["RL"] -= delta

    return desired_forces 

def compute_force_error(desired_forces, measured_forces):
    """
    計算 GRF 誤差
    """
    force_error = {}
    for leg in LEGS:
        desired = float(desired_forces[leg])
        measured = float(measured_forces[leg])
        force_error[leg] = desired - measured

    return force_error

def compute_foot_force_commands(force_error, Kf=0.6, force_limit=25.0, force_sign=1.0, support_legs=None):
    """
    根據 GRF 誤差計算 foot force commands
    """
    foot_force_cmds={}
    for leg in LEGS:
        if support_legs is not None and leg not in support_legs:
            fz_cmd = 0.0
        else:
            fz_cmd = Kf * force_error[leg]

            # 第一版先不允許負的垂直力
            fz_cmd = max(fz_cmd, 0.0)

            fz_cmd = np.clip(
                fz_cmd,
                0.0,
                force_limit,
            )

        foot_force_cmds[leg] = np.array([
            0.0,
            0.0,
            force_sign * fz_cmd,
        ])
    return foot_force_cmds

def leg_jtf_torque(model, data, leg, ids, f_world):
    """
    Jacobian 某腳的足端力轉成三個關節 torque

    f_world:
        世界座標的足端力np.array([0.0, 0.0, 5.0])

    return:
        tau_leg: 3 維 torque
    """
    jacp = np.zeros((3, model.nv)) #position
    jacr = np.zeros((3, model.nv)) #rotation
    site_id = ids["site"][leg]

    #mujoco jacobian
    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        site_id,
    )
    #找出leg的關節
    leg_index = LEGS.index(leg)
    leg_start = leg_index * 3
    leg_dofs = ids["qvel"][leg_start:leg_start + 3]

    J_leg = jacp[:, leg_dofs]

    tau_leg = J_leg.T @ f_world

    return tau_leg


def scale_leg_torque(tau, leg, scale):
    """
    torque 乘上一個比例
    """
    tau_scaled = tau.copy()

    leg_index = LEGS.index(leg)
    leg_start = leg_index * 3

    tau_scaled[leg_start:leg_start + 3] *= scale

    return tau_scaled

def smooth_q_des(q_des, q_des_raw, alpha=0.05):
    """
    讓 q_des 慢慢接近 q_des_raw，避免 torque PD 突然跳太大。
    """
    return q_des + alpha * (q_des_raw - q_des)

def update_z_offset_from_force_error(
    z_offset,
    desired_forces,
    measured_forces,
    swing_leg,
    support_legs,
):
    """
    desired GRF 和 measured GRF 自動更新 z_offset。

    """

    new_z_offset = z_offset.copy()

    # 每次更新最大變化量，避免突然跳太大
    max_step = 0.00004

    # z offset 限制
    z_up_limit = 0.010      # 腳最多往上收 10 mm
    z_down_limit = -0.006   # 支撐腳最多往下踩 6 mm

    # 1. swing leg 卸重
    swing_measured = measured_forces[swing_leg]
    swing_desired = desired_forces.get(swing_leg, 0.0)

    swing_error = swing_measured - swing_desired

    if swing_error > 0.0:
        dz = 0.000002 * swing_error
        dz = np.clip(dz, 0.0, max_step)
        new_z_offset[swing_leg] += dz

    # 2. support legs 補支撐
    for leg in support_legs:
        desired = desired_forces[leg]
        measured = measured_forces[leg]

        support_error = desired - measured

        if support_error > 0.0:
            dz = -0.000002 * support_error
            dz = np.clip(dz, -max_step, 0.0)
            new_z_offset[leg] += dz

    # 3. 限制 z_offset 範圍
    for leg in LEGS:
        new_z_offset[leg] = float(
            np.clip(
                new_z_offset[leg],
                z_down_limit,
                z_up_limit,
            )
        )

    return new_z_offset

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)

            mujoco.mj_step(model, data)
            v.sync()

            step += 1
if __name__ == "__main__":
    main()