"""
model_probe.py

用途：
    從 MuJoCo A1 模型自動讀出 Level 2 卸重控制需要的幾何、sensor、actuator、CoM、baseline force。

用法：
    python3 model_probe.py /path/to/scene_torque.xml

建議：
    先用你實際 main_3D.py 載入的 XML 路徑跑，不要只用備份 XML。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

LEGS = ["FR", "FL", "RR", "RL"]
FOOT_SITE = {
    "FR": "fr_touch_site",
    "FL": "fl_touch_site",
    "RR": "rr_touch_site",
    "RL": "rl_touch_site",
}
HIP_BODY = {
    "FR": "FR_hip",
    "FL": "FL_hip",
    "RR": "RR_hip",
    "RL": "RL_hip",
}
TOUCH_SENSOR = {
    "FR": "fr_touch",
    "FL": "fl_touch",
    "RR": "rr_touch",
    "RL": "rl_touch",
}
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


def np_list(x, ndigits=6):
    return np.round(np.asarray(x, dtype=float), ndigits).tolist()


def mj_name(model, obj_type, idx):
    name = mujoco.mj_id2name(model, obj_type, idx)
    return name if name is not None else ""


def world_to_body_xy(data, body_id: int, p_world):
    """World point -> body local xy."""
    p_world = np.asarray(p_world, dtype=float)
    p0 = data.xpos[body_id]
    R = data.xmat[body_id].reshape(3, 3)  # local -> world
    p_body = R.T @ (p_world - p0)
    return p_body[:2]


def world_to_body_xyz(data, body_id: int, p_world):
    p_world = np.asarray(p_world, dtype=float)
    p0 = data.xpos[body_id]
    R = data.xmat[body_id].reshape(3, 3)
    return R.T @ (p_world - p0)


def get_sensor_scalar(model, data, sensor_name: str) -> float:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    adr = int(model.sensor_adr[sid])
    return float(data.sensordata[adr])


def get_actuator_info(model):
    info = {}
    for name in ACTUATOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            info[name] = {"exists": False}
            continue
        info[name] = {
            "exists": True,
            "id": int(aid),
            "ctrlrange": np_list(model.actuator_ctrlrange[aid]),
            "forcerange": np_list(model.actuator_forcerange[aid]),
            "gear": np_list(model.actuator_gear[aid]),
            "trntype": int(model.actuator_trntype[aid]),
            "gaintype": int(model.actuator_gaintype[aid]),
            "biastype": int(model.actuator_biastype[aid]),
        }
    return info


def get_joint_info(model, data):
    out = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            out[name] = {"exists": False}
            continue
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        out[name] = {
            "exists": True,
            "id": int(jid),
            "qposadr": qadr,
            "qveladr": dadr,
            "qpos": float(data.qpos[qadr]),
            "qvel": float(data.qvel[dadr]),
            "range": np_list(model.jnt_range[jid]),
        }
    return out


def read_model(xml_path: Path, settle_steps: int = 1000):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)

    # 保持 home ctrl，讓接觸力進入穩態。若 XML 是 motor torque 模型，這裡的結果也會暴露出來。
    ctrl_home = data.ctrl.copy()
    for _ in range(settle_steps):
        data.ctrl[:] = ctrl_home
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id < 0:
        raise RuntimeError("找不到 body: trunk")

    com_world = np.array(data.subtree_com[trunk_id], dtype=float)
    trunk_world = np.array(data.xpos[trunk_id], dtype=float)
    com_from_trunk_body = world_to_body_xyz(data, trunk_id, com_world)

    hip_xyz_body = {}
    foot_xyz_body = {}
    foot_xy_body = {}
    foot_world = {}

    for leg in LEGS:
        hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HIP_BODY[leg])
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITE[leg])
        if hip_id < 0:
            raise RuntimeError(f"找不到 body: {HIP_BODY[leg]}")
        if site_id < 0:
            raise RuntimeError(f"找不到 site: {FOOT_SITE[leg]}")

        hip_xyz_body[leg] = np_list(world_to_body_xyz(data, trunk_id, data.xpos[hip_id]))
        foot_xyz = world_to_body_xyz(data, trunk_id, data.site_xpos[site_id])
        foot_xyz_body[leg] = np_list(foot_xyz)
        foot_xy_body[leg] = np_list(foot_xyz[:2])
        foot_world[leg] = np_list(data.site_xpos[site_id])

    forces = {}
    for leg in LEGS:
        sensor_name = TOUCH_SENSOR[leg]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            forces[leg] = None
        else:
            forces[leg] = get_sensor_scalar(model, data, sensor_name)

    sensors = []
    for i in range(model.nsensor):
        sensors.append({
            "id": int(i),
            "name": mj_name(model, mujoco.mjtObj.mjOBJ_SENSOR, i),
            "adr": int(model.sensor_adr[i]),
            "dim": int(model.sensor_dim[i]),
            "type": int(model.sensor_type[i]),
        })

    result = {
        "xml_path": str(xml_path),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nsensor": int(model.nsensor),
        "timestep": float(model.opt.timestep),
        "total_body_mass": float(np.sum(model.body_mass)),
        "trunk_world": np_list(trunk_world),
        "trunk_quat_qpos_wxyz": np_list(data.qpos[3:7]),
        "com_world": np_list(com_world),
        "com_from_trunk_body": np_list(com_from_trunk_body),
        "hip_xyz_body": hip_xyz_body,
        "foot_xyz_body": foot_xyz_body,
        "foot_xy_body": foot_xy_body,
        "foot_world": foot_world,
        "baseline_forces_after_settle": {k: None if v is None else round(v, 6) for k, v in forces.items()},
        "qpos": np_list(data.qpos),
        "qvel": np_list(data.qvel),
        "ctrl_home_after_settle": np_list(ctrl_home),
        "joint_info": get_joint_info(model, data),
        "actuator_info": get_actuator_info(model),
        "sensors": sensors,
    }
    return result


def main():
    if len(sys.argv) >= 2:
        xml_path = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"
    else:
        xml_path = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"XML 不存在: {xml_path}")

    result = read_model(xml_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = xml_path.parent / "model_probe_output.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已輸出: {out_path}")


if __name__ == "__main__":
    main()