import json

import mujoco
import numpy as np
from mujoco import viewer
from pathlib import Path
from grf_redistributor import compute_grf_redistribution as compute_cop_grf_redistribution

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


def np_round_list(x, ndigits=6):
    return np.round(np.asarray(x, dtype=float), ndigits).tolist()


def np_round_dict(d, ndigits=6):
    return {k: np_round_list(v, ndigits) for k, v in d.items()}


def get_trunk_frame(model, data):
    trunk_id = must_find_id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    trunk_pos_world = data.xpos[trunk_id].copy()
    trunk_rot_world = data.xmat[trunk_id].reshape(3, 3).copy()
    return trunk_id, trunk_pos_world, trunk_rot_world


def world_to_body(data, body_id, p_world):
    body_pos_world = data.xpos[body_id].copy()
    body_rot_world = data.xmat[body_id].reshape(3, 3).copy()
    return body_rot_world.T @ (np.asarray(p_world, dtype=float) - body_pos_world)


def get_foot_xyz_body(model, data, ids):
    trunk_id, trunk_pos_world, trunk_rot_world = get_trunk_frame(model, data)
    foot_xyz_body = {}

    for leg in LEGS:
        site_id = ids["site"][leg]
        foot_pos_world = data.site_xpos[site_id].copy()
        foot_xyz_body[leg] = trunk_rot_world.T @ (foot_pos_world - trunk_pos_world)

    return foot_xyz_body


def get_hip_xyz_body(model, data):
    trunk_id, trunk_pos_world, trunk_rot_world = get_trunk_frame(model, data)
    hip_xyz_body = {}

    for leg in LEGS:
        hip_id = must_find_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
        hip_pos_world = data.xpos[hip_id].copy()
        hip_xyz_body[leg] = trunk_rot_world.T @ (hip_pos_world - trunk_pos_world)

    return hip_xyz_body


def get_com_info(model, data):
    trunk_id, trunk_pos_world, trunk_rot_world = get_trunk_frame(model, data)
    com_world = data.subtree_com[trunk_id].copy()
    com_from_trunk_body = trunk_rot_world.T @ (com_world - trunk_pos_world)
    return com_world, com_from_trunk_body


def get_joint_info_dict(model, data, ids):
    q, qd = read_joint_states(data, ids)
    joint_info = {}

    for i, name in enumerate(JOINT_NAMES):
        joint_id = must_find_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_info[name] = {
            "qposadr": int(ids["qpos"][i]),
            "qveladr": int(ids["qvel"][i]),
            "q": float(q[i]),
            "qd": float(qd[i]),
            "range": np_round_list(model.jnt_range[joint_id]),
        }

    return joint_info


def torque_array_to_dict(tau):
    tau = np.asarray(tau, dtype=float)
    out = {}
    for i, actuator_name in enumerate(ACTUATOR_NAMES):
        out[actuator_name] = float(tau[i])
    return out


def force_cmds_to_dict(foot_force_cmds):
    return {
        leg: np_round_list(foot_force_cmds[leg])
        for leg in LEGS
    }


def build_live_probe_snapshot(
    model,
    data,
    ids,
    step,
    q,
    qd,
    q_des,
    qd_des,
    forces,
    desired_forces,
    force_error,
    foot_force_cmds,
    tau_stand,
    tau_grf,
    tau_total,
    body_shift_cmd,
    swing_leg,
    support_legs,
    grf_debug,
    swing_scale,
    Kp,
    Kd,
    Kf,
    tau_limit,
):
    foot_xyz_body = get_foot_xyz_body(model, data, ids)
    foot_xy_body = {leg: foot_xyz_body[leg][:2] for leg in LEGS}
    hip_xyz_body = get_hip_xyz_body(model, data)
    com_world, com_from_trunk_body = get_com_info(model, data)
    trunk_id, trunk_pos_world, _ = get_trunk_frame(model, data)

    tau_total = np.asarray(tau_total, dtype=float)
    tau_stand = np.asarray(tau_stand, dtype=float)
    tau_grf = np.asarray(tau_grf, dtype=float)

    snapshot = {
        "step": int(step),
        "time": float(data.time),
        "xml_model": "scene_torque.xml",
        "dt": float(model.opt.timestep),
        "total_body_mass": float(np.sum(model.body_mass)),
        "controller": {
            "mode": "torque_pd_plus_grf",
            "Kp": float(Kp),
            "Kd": float(Kd),
            "Kf": float(Kf),
            "tau_limit": float(tau_limit),
            "swing_leg": swing_leg,
            "support_legs": list(support_legs),
            "swing_scale": float(swing_scale),
        },
        "trunk_world": np_round_list(trunk_pos_world),
        "trunk_qpos_xyz_quat_wxyz": np_round_list(data.qpos[:7]),
        "com_world": np_round_list(com_world),
        "com_from_trunk_body": np_round_list(com_from_trunk_body),
        "hip_xyz_body": np_round_dict(hip_xyz_body),
        "foot_xyz_body": np_round_dict(foot_xyz_body),
        "foot_xy_body": np_round_dict(foot_xy_body),
        "forces": {leg: float(forces[leg]) for leg in LEGS},
        "desired_forces": {leg: float(desired_forces[leg]) for leg in LEGS},
        "force_error": {leg: float(force_error[leg]) for leg in LEGS},
        "foot_force_cmds_world": force_cmds_to_dict(foot_force_cmds),
        "cop": {
            "measured_cop": np_round_list(grf_debug.get("measured_cop", np.zeros(2))),
            "target_cop": np_round_list(grf_debug.get("target_cop", np.zeros(2))),
            "cop_error": np_round_list(grf_debug.get("cop_error", np.zeros(2))),
        },
        "body_shift_cmd": np_round_list(body_shift_cmd),
        "joint": {
            "q": np_round_list(q),
            "qd": np_round_list(qd),
            "q_des": np_round_list(q_des),
            "qd_des": np_round_list(qd_des),
            "q_error": np_round_list(np.asarray(q_des) - np.asarray(q)),
            "qd_norm": float(np.linalg.norm(qd)),
            "joint_info": get_joint_info_dict(model, data, ids),
        },
        "torque": {
            "tau_stand": torque_array_to_dict(tau_stand),
            "tau_grf": torque_array_to_dict(tau_grf),
            "tau_total": torque_array_to_dict(tau_total),
            "max_abs_tau_stand": float(np.max(np.abs(tau_stand))),
            "max_abs_tau_grf": float(np.max(np.abs(tau_grf))),
            "max_abs_tau_total": float(np.max(np.abs(tau_total))),
            "saturated": bool(np.max(np.abs(tau_total)) >= tau_limit - 1e-6),
        },
    }

    return snapshot


def print_live_probe_summary(snapshot):
    forces = snapshot["forces"]
    torque = snapshot["torque"]
    cop = snapshot["cop"]

    print("\n========== LIVE STANDING PROBE ==========")
    print("step:", snapshot["step"], "time:", round(snapshot["time"], 3))
    print("trunk_world:", snapshot["trunk_world"])
    print("com_world:", snapshot["com_world"])
    print("com_from_trunk_body:", snapshot["com_from_trunk_body"])
    print("forces:", {leg: round(forces[leg], 3) for leg in LEGS})
    print("foot_xy_body:", snapshot["foot_xy_body"])
    print("hip_xyz_body:", snapshot["hip_xyz_body"])
    print("measured_cop:", cop["measured_cop"], "target_cop:", cop["target_cop"], "cop_error:", cop["cop_error"])
    print("body_shift_cmd:", snapshot["body_shift_cmd"])
    print("max_tau_stand:", round(torque["max_abs_tau_stand"], 3),
          "max_tau_grf:", round(torque["max_abs_tau_grf"], 3),
          "max_tau_total:", round(torque["max_abs_tau_total"], 3),
          "saturated:", torque["saturated"])
    print("qd_norm:", round(snapshot["joint"]["qd_norm"], 6))
    print("========================================\n")


def save_live_probe_snapshot(snapshot, output_path):
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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

    elif mode == "unload_RL":
        swing_leg = "RL"
        support_legs = ["FR", "FL", "RR"]

        rl_target = 15.0
        remaining_force = total_force - rl_target
        support_force = remaining_force / 3.0

        for leg in support_legs:
            desired_forces[leg] = support_force

        desired_forces[swing_leg] = rl_target

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

def compute_grf_torque(model, data, ids, foot_force_cmds):
    """
    leg_jtf 套用到四肢腳
    """
    tau_grf = np.zeros(12)
    for leg in LEGS:
        f_world = foot_force_cmds[leg]
        tau_leg = leg_jtf_torque(
            model,
            data,
            leg,
            ids,
            f_world,
        )

        leg_index = LEGS.index(leg)
        leg_start = leg_index * 3
        tau_grf[leg_start:leg_start + 3] = tau_leg

    return tau_grf

def scale_leg_torque(tau, leg, scale):
    """
    torque 乘上一個比例
    """
    tau_scaled = tau.copy()

    leg_index = LEGS.index(leg)
    leg_start = leg_index * 3

    tau_scaled[leg_start:leg_start + 3] *= scale

    return tau_scaled

def compute_body_shift_from_cop_error(cop_error, gain=0.2, max_shift=0.025):
    """
  CoP error 轉 body xy shift command
    """
    body_shift = gain * cop_error

    body_shift = np.clip(
        body_shift,
        -max_shift,
        max_shift,
    )

    return body_shift

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"

    print("XML path:", xml)

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)


    ids = get_ids(model)

    # 1. reset 到 home pose
    key_name = "home"
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        key_name,
    )

    if key_id < 0:
        raise RuntimeError("找不到 home keyframe")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # 2. 設定 home pose 為站立目標
    q_des = np.array([data.qpos[i] for i in ids["qpos"]])
    qd_des = np.zeros(12)

    # 3. torque PD 參數
    Kp = 100.0
    Kd = 3.0

    Kf = 0.8
    force_limit = 25.0
    force_sign = 1.0

    tau_limit = 33.5
    step = 0

    # Probe 設定：讓 MuJoCo 照常站著，定期把當下真實狀態印出並存成 JSON。
    probe_output_path = Path("standing_live_probe.json")
    probe_print_every = 500
    probe_save_every = 500
    probe_start_step = 1000

    # 4. 開啟 MuJoCo viewer
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)

            # 1. 讀目前關節狀態
            q, qd = read_joint_states(data, ids)

            # 2. 讀目前四腳受力 measured GRF
            forces = read_touch_forces(data, ids)
            foot_xy_body = get_foot_xy_body(
                model,
                data,
                ids,
            )

            # 3. 計算 desired GRF
            swing_leg = "RL"

            desired_forces, force_error, cop_error, grf_debug = compute_cop_grf_redistribution(
                swing_leg,
                forces,
                foot_xy_body,
            )
            support_legs = grf_debug["support_legs"]

            foot_force_cmds = compute_foot_force_commands(
                force_error,
                Kf=Kf,
                force_limit=force_limit,
                force_sign=force_sign,
                support_legs=support_legs,
            )

            tau_stand = compute_standing_torque(
                data,
                ids,
                q,
                qd,
                q_des,
                qd_des,
                Kp,
                Kd,
            )
            cop_error_norm = np.linalg.norm(cop_error)

            if cop_error_norm > 0.04:
                # CoP 還沒移到位，先不要卸 RL
                swing_scale = 1.0
            else:
                # CoP 接近 target 後，才開始卸 RL
                swing_scale = 0.7

            body_shift_cmd = compute_body_shift_from_cop_error(
                cop_error,
                gain=0.2,
                max_shift=0.025,
            )

            tau_stand = scale_leg_torque(
                tau_stand,
                swing_leg,
                scale=swing_scale,
            )

            tau_grf = compute_grf_torque(
                model,
                data,
                ids,
                foot_force_cmds,
            )

            tau_total = tau_stand + tau_grf

            # 9. 寫入 motor actuator
            tau_total = write_torque(
                data,
                ids,
                tau_total,
                tau_limit,
            )

            # 10. debug：短訊息保留，避免你看不到控制是否正常。
            if step % 100 == 0:
                print(
                    "step:", step,
                    "z:", round(data.qpos[2], 3),
                    "measured:", {leg: round(force, 1) for leg, force in forces.items()},
                    "desired:", {leg: round(force, 1) for leg, force in desired_forces.items()},
                    "error:", {leg: round(err, 1) for leg, err in force_error.items()},
                    "Fz_cmd:", {leg: round(float(cmd[2]), 2) for leg, cmd in foot_force_cmds.items()},
                    "max_tau_grf:", round(float(np.max(np.abs(tau_grf))), 2),
                    "max_tau:", round(float(np.max(np.abs(tau_total))), 2),
                    "cop:", np.round(grf_debug["measured_cop"], 3).tolist(),
                    "target_cop:", np.round(grf_debug["target_cop"], 3).tolist(),
                    "cop_error:", np.round(cop_error, 3).tolist(),
                    "swing_scale:", round(swing_scale, 2),
                    "cop_err_norm:", round(float(cop_error_norm), 3),
                    "body_shift_cmd:", np.round(body_shift_cmd, 3).tolist(),
                )

            # 11. live probe：站穩一段時間後，輸出 Level 2 需要的正式模型/狀態資料。
            if step >= probe_start_step and step % probe_save_every == 0:
                snapshot = build_live_probe_snapshot(
                    model=model,
                    data=data,
                    ids=ids,
                    step=step,
                    q=q,
                    qd=qd,
                    q_des=q_des,
                    qd_des=qd_des,
                    forces=forces,
                    desired_forces=desired_forces,
                    force_error=force_error,
                    foot_force_cmds=foot_force_cmds,
                    tau_stand=tau_stand,
                    tau_grf=tau_grf,
                    tau_total=tau_total,
                    body_shift_cmd=body_shift_cmd,
                    swing_leg=swing_leg,
                    support_legs=support_legs,
                    grf_debug=grf_debug,
                    swing_scale=swing_scale,
                    Kp=Kp,
                    Kd=Kd,
                    Kf=Kf,
                    tau_limit=tau_limit,
                )

                save_live_probe_snapshot(snapshot, probe_output_path)

                if step % probe_print_every == 0:
                    print_live_probe_summary(snapshot)
                    print("probe saved:", probe_output_path.resolve())

            mujoco.mj_step(model, data)
            v.sync()

            step += 1
if __name__ == "__main__":
    main()