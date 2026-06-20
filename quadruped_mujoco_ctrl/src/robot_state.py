import mujoco
import numpy as np

import config as cfg


class MujocoModelIds:
    """
    存 MuJoCo 裡面常用物件的 ID。

    為什麼要做這個 class？
    因為 mujoco.mj_name2id() 不應該每個 loop 都一直查。
    程式一開始查一次，之後直接用 ID 讀資料。
    """

    def __init__(
        self,
        qpos_ids,
        qvel_ids,
        actuator_ids,
        site_ids,
        touch_adrs,
        trunk_body_id,  
    ):
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.actuator_ids = actuator_ids
        self.site_ids = site_ids
        self.touch_adrs = touch_adrs
        self.trunk_body_id = trunk_body_id


class RobotState:
    """
    存目前這一瞬間的機器人狀態。

    這個 class 只存資料，不做控制。
    """

    def __init__(
        self,
        time,
        q,
        qd,
        trunk_pos_world,
        trunk_quat_wxyz,
        trunk_rot_world,
        com_world,
        com_body,
        com_xy_body,
        foot_world,
        foot_xyz_body,
        foot_xy_body,
        forces,
        force_vector,
        total_force,
    ):
        self.time = time

        self.q = q
        self.qd = qd

        self.trunk_pos_world = trunk_pos_world
        self.trunk_quat_wxyz = trunk_quat_wxyz
        self.trunk_rot_world = trunk_rot_world

        self.com_world = com_world
        self.com_body = com_body
        self.com_xy_body = com_xy_body

        self.foot_world = foot_world
        self.foot_xyz_body = foot_xyz_body
        self.foot_xy_body = foot_xy_body

        self.forces = forces
        self.force_vector = force_vector
        self.total_force = total_force


def must_find_id(model, obj_type, name):
    """
    用名字找 MuJoCo ID。

    找不到就直接報錯，不要讓程式默默繼續跑。
    """

    obj_id = mujoco.mj_name2id(model, obj_type, name)

    if obj_id < 0:
        raise RuntimeError(f"找不到 MuJoCo 物件: {name}")

    return int(obj_id)


def build_mujoco_ids(model):
    """
    建立 MuJoCo ID 對照表。

    return:
        MujocoModelIds
    """

    qpos_ids = []
    qvel_ids = []
    actuator_ids = []

    for joint_name, actuator_name in zip(
        cfg.JOINT_NAMES,
        cfg.ACTUATOR_NAMES, 
    ):
        joint_id = must_find_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        actuator_id = must_find_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )

        qpos_id = int(model.jnt_qposadr[joint_id])
        qvel_id = int(model.jnt_dofadr[joint_id])

        qpos_ids.append(qpos_id)
        qvel_ids.append(qvel_id)
        actuator_ids.append(int(actuator_id))

    site_ids = {}

    for leg in cfg.LEGS:
        site_name = cfg.FOOT_SITE_NAMES[leg]

        site_id = must_find_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        site_ids[leg] = site_id

    touch_adrs = {}

    for leg in cfg.LEGS:
        sensor_name = cfg.TOUCH_SENSOR_NAMES[leg]

        sensor_id = must_find_id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )

        sensor_adr = int(model.sensor_adr[sensor_id])
        touch_adrs[leg] = sensor_adr

    trunk_body_id = must_find_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "trunk",
    )

    ids = MujocoModelIds(
        qpos_ids=qpos_ids,
        qvel_ids=qvel_ids,
        actuator_ids=actuator_ids,
        site_ids=site_ids,
        touch_adrs=touch_adrs,
        trunk_body_id=trunk_body_id,
    )

    return ids


def world_to_body_xyz(trunk_pos_world, trunk_rot_world, point_world):
    """
    world frame 的點轉成 trunk/body frame。

    MuJoCo 給 foot site 的位置通常是 world 座標。
    但 QP 和 IK 比較常用 body frame。
    """

    point_world = np.asarray(point_world, dtype=float)

    point_body = trunk_rot_world.T @ (
        point_world - trunk_pos_world
    )

    return point_body


def read_joint_state(data, ids):
    """
    讀 12 維關節角度 q 和關節速度 qd。

    順序由 cfg.JOINT_NAMES 決定：
    FR, FL, RR, RL
    每腳 hip, thigh, calf
    """

    q = np.array(
        [data.qpos[i] for i in ids.qpos_ids],
        dtype=float,
    )

    qd = np.array(
        [data.qvel[i] for i in ids.qvel_ids],
        dtype=float,
    )

    if q.shape != (12,):
        raise RuntimeError(f"q shape 錯誤，目前是 {q.shape}")

    if qd.shape != (12,):
        raise RuntimeError(f"qd shape 錯誤，目前是 {qd.shape}")

    return q, qd


def read_touch_forces(data, ids):
    """
    讀四腳 touch sensor force。

    return:
        forces = {
            "FR": ...,
            "FL": ...,
            "RR": ...,
            "RL": ...,
        }
    """

    forces = {}

    for leg in cfg.LEGS:
        adr = ids.touch_adrs[leg]
        force = float(data.sensordata[adr])
        forces[leg] = force

    return forces


def read_foot_positions(
    data,
    ids,
    trunk_pos_world,
    trunk_rot_world,
):
    """
    讀四腳 foot site 位置。

    return:
        foot_world
        foot_xyz_body
        foot_xy_body
    """

    foot_world = {}
    foot_xyz_body = {}
    foot_xy_body = {}

    for leg in cfg.LEGS:
        site_id = ids.site_ids[leg]

        p_world = np.array(
            data.site_xpos[site_id],
            dtype=float,
            copy=True,
        )

        p_body = world_to_body_xyz(
            trunk_pos_world,
            trunk_rot_world,
            p_world,
        )

        foot_world[leg] = p_world
        foot_xyz_body[leg] = p_body
        foot_xy_body[leg] = p_body[:2].copy()

    return foot_world, foot_xyz_body, foot_xy_body


def read_robot_state(model, data, ids):
    """
    從 MuJoCo model/data 讀出目前機器人狀態。

    注意：
    這個函式不呼叫 mujoco.mj_step()
    這個函式不呼叫 mujoco.mj_forward()
    它只讀資料。
    """

    q, qd = read_joint_state(data, ids)

    trunk_pos_world = np.array(
        data.xpos[ids.trunk_body_id],
        dtype=float,
        copy=True,
    )

    trunk_rot_world = np.array(
        data.xmat[ids.trunk_body_id].reshape(3, 3),
        dtype=float,
        copy=True,
    )

    trunk_quat_wxyz = np.array(
        data.qpos[3:7],
        dtype=float,
        copy=True,
    )

    com_world = np.array(
        data.subtree_com[ids.trunk_body_id],
        dtype=float,
        copy=True,
    )

    com_body = world_to_body_xyz(
        trunk_pos_world,
        trunk_rot_world,
        com_world,
    )

    com_xy_body = com_body[:2].copy()

    foot_world, foot_xyz_body, foot_xy_body = read_foot_positions(
        data,
        ids,
        trunk_pos_world,
        trunk_rot_world,
    )

    forces = read_touch_forces(data, ids)

    force_vector = np.array(
        [forces[leg] for leg in cfg.LEGS],
        dtype=float,
    )

    total_force = float(np.sum(force_vector))

    state = RobotState(
        time=float(data.time),
        q=q,
        qd=qd,
        trunk_pos_world=trunk_pos_world,
        trunk_quat_wxyz=trunk_quat_wxyz,
        trunk_rot_world=trunk_rot_world,
        com_world=com_world,
        com_body=com_body,
        com_xy_body=com_xy_body,
        foot_world=foot_world,
        foot_xyz_body=foot_xyz_body,
        foot_xy_body=foot_xy_body,
        forces=forces,
        force_vector=force_vector,
        total_force=total_force,
    )

    return state