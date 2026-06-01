import mujoco
import numpy as np
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

LEG_SIDE_SIGN = {
    "FR": -1.0,
    "RR": -1.0,
    "FL":  1.0,
    "RL":  1.0,
}

class RobotState:
    def __init__(self, q, qd, forces, foot_xy_body, foot_xyz_body, hip_xyz_body, trunk_pos_world, trunk_rot_world):
        self.q = q
        self.qd = qd
        self.forces = forces
        self.foot_xy_body = foot_xy_body
        self.foot_xyz_body = foot_xyz_body
        self.hip_xyz_body = hip_xyz_body
        self.trunk_pos_world = trunk_pos_world
        self.trunk_rot_world = trunk_rot_world

class RobotInterface:
    def __init__(self, xml_path, torque_limit=33.5):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.ids = get_ids(self.model)
        self.torque_limit = torque_limit

        mujoco.mj_forward(self.model, self.data)
    
    def read_state(self):
        q, qd = read_joint_states(self.data, self.ids)

        forces = read_touch_forces(self.data, self.ids)

        trunk_pos_world, trunk_rot_world = get_trunk_pose_world(
            self.model,
            self.data,
        )
        
        foot_xy_body = get_foot_xy_body(
            self.model,
            self.data,
            self.ids,
        )

        foot_xyz_body = get_foot_xyz_body(
            self.model,
            self.data,
            self.ids,
        )

        hip_xyz_body = get_hip_xyz_body(
            self.model,
            self.data,
        )


        state = RobotState(
            q=q,
            qd=qd,
            forces=forces,
            foot_xy_body=foot_xy_body,
            foot_xyz_body=foot_xyz_body,
            hip_xyz_body=hip_xyz_body,
            trunk_pos_world=trunk_pos_world,
            trunk_rot_world=trunk_rot_world,
        )

        return state
    
    def write_torque(self, tau):
        tau_safe = write_torque(
            self.data,
            self.ids,
            tau,
            tau_limit=self.torque_limit,
        )

        return tau_safe

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

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

def get_trunk_pose_world(model, data):
    trunk_id = must_find_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "trunk",
    )

    trunk_pos_world = data.xpos[trunk_id].copy()
    trunk_rot_world = data.xmat[trunk_id].reshape(3, 3).copy()

    return trunk_pos_world, trunk_rot_world

def get_foot_xy_body(model, data, ids, trunk_pos_world, trunk_rot_world):
    """
    取得四隻腳 foot site 在 body/trunk frame 下的 xy 位置
    """
    foot_xy_body = {}

    for leg in LEGS:
        site_id = ids["site"][leg]
        foot_pos_world = data.site_xpos[site_id].copy()

        foot_pos_body = trunk_rot_world.T @ (
            foot_pos_world - trunk_pos_world
        )

        foot_xy_body[leg] = foot_pos_body[:2].copy()

    return foot_xy_body

def get_foot_xyz_body(model, data, ids, trunk_pos_world, trunk_rot_world):
    foot_xyz_body = {}

    for leg in LEGS:
        site_id = ids["site"][leg]
        foot_pos_world = data.site_xpos[site_id].copy()

        foot_pos_body = trunk_rot_world.T @ (
            foot_pos_world - trunk_pos_world
        )

        foot_xyz_body[leg] = foot_pos_body.copy()

    return foot_xyz_body

def get_hip_xyz_body(model, data, trunk_pos_world, trunk_rot_world):
    """
    取得四隻腳 hip joint anchor 在 body/trunk frame 下的 xyz 位置
    """

    hip_xyz_body = {}

    for leg in LEGS:
        hip_joint_name = f"{leg}_hip_joint"

        hip_joint_id = must_find_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            hip_joint_name,
        )

        hip_pos_world = data.xanchor[hip_joint_id].copy()

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

def write_torque(data, ids, tau, tau_limit=33.5):

    tau = np.clip(tau, -tau_limit, tau_limit)
    for i, actuator_id in enumerate(ids["actuator"]):
        data.ctrl[actuator_id] = tau[i]
    return tau


    
    