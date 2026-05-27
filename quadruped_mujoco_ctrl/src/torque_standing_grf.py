import mujoco
import numpy as np
from mujoco import viewer
from pathlib import Path

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
        delta = 8.0

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

def compute_foot_force_commands(force_error, Kf=0.6, force_limit=15.0, force_sign=1.0):
    """
    根據 GRF 誤差計算 foot force commands
    """
    foot_force_cmds={}
    for leg in LEGS:
        fz_cmd = Kf * force_error[leg]

        fz_cmd = np.clip(
            fz_cmd,
            -force_limit,
            force_limit,
        )

        foot_force_cmds[leg] = np.array([
            0.0,
            0.0,
            force_sign * fz_cmd,
        ])

    return foot_force_cmds

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"

    print("XML path:", xml)

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    print("model.nq =", model.nq)
    print("model.nv =", model.nv)
    print("model.nu =", model.nu)

    ids = get_ids(model)

    print("\n=== ids 檢查 ===")
    print("qpos ids:", ids["qpos"])
    print("qvel ids:", ids["qvel"])
    print("actuator ids:", ids["actuator"])
    print("site ids:", ids["site"])
    print("touch adrs:", ids["touch_adr"])

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
    tau_limit = 33.5

    step = 0

    # 4. 開啟 MuJoCo viewer
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)

            # 5. 讀目前關節角與關節速度
            q = np.array([data.qpos[i] for i in ids["qpos"]])
            qd = np.array([data.qvel[i] for i in ids["qvel"]])

            # 6. joint PD torque
            tau_pd = Kp * (q_des - q) + Kd * (qd_des - qd)

            # 7. bias / gravity compensation
            tau_bias = np.array([data.qfrc_bias[i] for i in ids["qvel"]])

            # 8. 合成 torque
            tau = tau_bias + tau_pd
            tau = np.clip(tau, -tau_limit, tau_limit)

            # 9. 寫入 motor actuator
            for i, actuator_id in enumerate(ids["actuator"]):
                data.ctrl[actuator_id] = tau[i]

            # 10. 讀四腳 touch force
            forces = read_touch_forces(data, ids)
            desired_forces = compute_desired_grf(
                forces,
                mode="shift_to_FR",
            )

            force_error = compute_force_error(
                desired_forces,
                forces,
            )
            foot_force_cmds = compute_foot_force_commands(
                force_error,
                Kf=0.6,
                force_limit=15.0,
                force_sign=1.0,
            )

            if step % 100 == 0:
                print(
                    "z:", round(data.qpos[2], 3),
                    "measured:", {leg: round(force, 1) for leg, force in forces.items()},
                    "desired:", {leg: round(force, 1) for leg, force in desired_forces.items()},
                    "error:", {leg: round(err, 1) for leg, err in force_error.items()},
                    "Fz_cmd:", {leg: round(float(cmd[2]), 2) for leg, cmd in foot_force_cmds.items()},
                    "max_tau:", round(float(np.max(np.abs(tau))), 2),
                )

            mujoco.mj_step(model, data)
            v.sync()

            step += 1
if __name__ == "__main__":
    main()