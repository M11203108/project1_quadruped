import mujoco
import numpy as np
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from mujoco import viewer
from kinematics import backward_kinematics_3d, forward_kinematics_3d
from unload_controller import check_com_margin, get_support_legs
from grf_redistributor import compute_grf_redistribution
from cmd_vel_sub import CmdVelSubscriber
from pathlib import Path
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32


x_home, z_home = 0.0, -0.24864398730826576
y_fr_home = -0.08505
y_fl_home = +0.08505
y_rr_home = -0.08505
y_rl_home = +0.08505
abd_angle, hip_angle, knee_angle = 0.0, 0.9, -1.8
h, hu, hl = 0.08505, 0.2, 0.2
lift_height = 0.04
k_lin = 1.0
k_yaw = 0.25
T = 0.80  # 每 1 秒切換一次
BASE_DIR = Path(__file__).resolve().parents[2]
xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene.xml"
LEGS = ["FR", "FL", "RR", "RL"]

GAIT_ORDER = ["FR", "RL", "FL", "RR"]
# Load model
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
print("nsensor =", model.nsensor)

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

def set_leg_ctrl_3d(ctrl, abd_id, thigh_id, calf_id, x, y, z, h, hu, hl, side_angle, ctrl_range):
    abd_angle, thigh_angle, calf_angle = backward_kinematics_3d(x, y, z, h, hu, hl, side_angle)
    abd_angle = np.clip(abd_angle, ctrl_range[abd_id, 0], ctrl_range[abd_id, 1])
    thigh_angle = np.clip(thigh_angle, ctrl_range[thigh_id, 0], ctrl_range[thigh_id, 1])
    calf_angle = np.clip(calf_angle, ctrl_range[calf_id, 0], ctrl_range[calf_id, 1])

    ctrl[abd_id] = abd_angle
    ctrl[thigh_id] = thigh_angle
    ctrl[calf_id] = calf_angle

def detect_foot_contact(data, foot_body_ids, model):

    contacts = {
        "FR": False,
        "FL": False,
        "RR": False,
        "RL": False
    }
    for i in range(data.ncon):

        con = data.contact[i]
        g1 = int(con.geom[0])
        g2 = int(con.geom[1])
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        for leg, body_id in foot_body_ids.items():
            if b1 == body_id or b2 == body_id:
                contacts[leg] = True

    return contacts

def get_com_xy(model, data):
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    com_xyz = data.subtree_com[trunk_id]
    return np.array([float(com_xyz[0]), float(com_xyz[1])])

def publish_imu(gyro_node, gyro_adr, gyro_dim, imu_pub, acc_adr, acc_dim):
    
    gyro = np.array(data.sensordata[gyro_adr:gyro_adr + gyro_dim], copy=True)
    acc  = np.array(data.sensordata[acc_adr:acc_adr + acc_dim], copy=True)

    msg = Imu()
    msg.header.stamp = gyro_node.get_clock().now().to_msg()
    msg.header.frame_id = "imu_link"

    msg.orientation_covariance[0] = -1.0
    msg.angular_velocity.x = float(gyro[0])
    msg.angular_velocity.y = float(gyro[1])
    msg.angular_velocity.z = float(gyro[2])

    msg.linear_acceleration.x = float(acc[0])
    msg.linear_acceleration.y = float(acc[1])
    msg.linear_acceleration.z = float(acc[2])

    imu_pub.publish(msg)

def publish_touch_sensor(touch_pubs ,data, fl_touch_adr, fr_touch_adr, rr_touch_adr, rl_touch_adr):

    forces= {
        "FR": float(data.sensordata[fr_touch_adr]),
        "FL": float(data.sensordata[fl_touch_adr]),
        "RR": float(data.sensordata[rr_touch_adr]),
        "RL": float(data.sensordata[rl_touch_adr]),
    }
    for leg, pub in touch_pubs.items():
        msg = Float32()
        msg.data = forces[leg]
        pub.publish(msg)

    return forces

def publish_base_pose(pose_pub, pose_node, twist_pub):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = pose_node.get_clock().now().to_msg()
    pose_msg.header.frame_id = "world"

    pose_msg.pose.position.x = float(data.qpos[0])
    pose_msg.pose.position.y = float(data.qpos[1])
    pose_msg.pose.position.z = float(data.qpos[2])

    pose_msg.pose.orientation.w = float(data.qpos[3])
    pose_msg.pose.orientation.x = float(data.qpos[4])
    pose_msg.pose.orientation.y = float(data.qpos[5])
    pose_msg.pose.orientation.z = float(data.qpos[6])

    twist_msg = TwistStamped()
    twist_msg.header.stamp = pose_node.get_clock().now().to_msg()
    twist_msg.header.frame_id = "world"

    twist_msg.twist.linear.x = float(data.qvel[0])
    twist_msg.twist.linear.y = float(data.qvel[1])
    twist_msg.twist.linear.z = float(data.qvel[2])

    twist_msg.twist.angular.x = float(data.qvel[3])
    twist_msg.twist.angular.y = float(data.qvel[4])
    twist_msg.twist.angular.z = float(data.qvel[5])

    pose_pub.publish(pose_msg)
    twist_pub.publish(twist_msg)

def publish_joint_states(joint_node, joint_pub, data):
    msg = JointState()
    msg.header.stamp = joint_node.get_clock().now().to_msg()
    msg.name = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    msg.position = data.qpos[7:19].tolist()
    msg.velocity = data.qvel[6:18].tolist()
    msg.effort = []

    joint_pub.publish(msg)

def publish_foot_contacts(contacts, contact_pubs):

    for leg in contact_pubs:
        pub = contact_pubs[leg]
        msg = Bool()
        msg.data = contacts[leg]

        pub.publish(msg)

def init_walk_state():
    """
    初始化走路控制器狀態
    控制器記憶
    """
    state = {
        # 控制階段
        "phase": "STAND",

        "phase_time": 0.0,

        # 抬腳順序
        "swing_index": 0,

        # 抬腳
        "swing_leg": GAIT_ORDER[0],

        # 身體支撐腳控制
        "body_shift": np.array([0.0, 0.0]),
        "target_body_shift": np.array([0.0, 0.0]),

        # z 微調支撐力分配
        "z_offset": {
            "FR": 0.0,
            "FL": 0.0,
            "RR": 0.0,
            "RL": 0.0,
        },

        # 每隻腳的預抬高度
        "pre_lift": {
            "FR": 0.0,
            "FL": 0.0,
            "RR": 0.0,
            "RL": 0.0,
        },

        # 腳前後位移
        "step_x": {
            "FR": 0.0,
            "FL": 0.0,
            "RR": 0.0,
            "RL": 0.0,
        },

        "ready_timer": 0.0,

        "debug": {},
    }

    return state

def main():
    rclpy.init()
    cmd_node = CmdVelSubscriber()

    pub_imu_node = rclpy.create_node("imu_publisher")
    imu_pub = pub_imu_node.create_publisher(Imu, "imu/data_raw", 10)

    pose_pub = pub_imu_node.create_publisher(PoseStamped,  "/a1/base/ground_truth_pose", 10)
    twist_pub = pub_imu_node.create_publisher(TwistStamped, "/a1/base/ground_truth_twist", 10)
    
    pub_touch_node = rclpy.create_node("touch_sensor_publisher")
    touch_pubs = {
        "FR": pub_touch_node.create_publisher(Float32, "/a1/touch/fr", 10),
        "FL": pub_touch_node.create_publisher(Float32, "/a1/touch/fl", 10),
        "RR": pub_touch_node.create_publisher(Float32, "/a1/touch/rr", 10),
        "RL": pub_touch_node.create_publisher(Float32, "/a1/touch/rl", 10),
    }
    #detect foot contact
    foot_body_ids ={
        "FR": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf"),
        "FL": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FL_calf"),
        "RR": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "RR_calf"),
        "RL": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "RL_calf"),
    }

    contact_pubs ={
        "FR": pub_imu_node.create_publisher(Bool, "/a1/foot_contact/fr", 10),
        "FL": pub_imu_node.create_publisher(Bool, "/a1/foot_contact/fl", 10),
        "RR": pub_imu_node.create_publisher(Bool, "/a1/foot_contact/rr", 10),
        "RL": pub_imu_node.create_publisher(Bool, "/a1/foot_contact/rl", 10),
    }
    

    pub_joint_node = rclpy.create_node("joint_state_publisher")
    joint_pub = pub_joint_node.create_publisher(JointState, "joint_states",10)

    #imu sensor adr and dim
    gyro_sensor = model.sensor("imu_gyro")
    acc_sensor  = model.sensor("imu_acc")
    gyro_adr = int(np.asarray(gyro_sensor.adr).item())
    gyro_dim = int(np.asarray(gyro_sensor.dim).item())

    acc_adr = int(np.asarray(acc_sensor.adr).item())
    acc_dim = int(np.asarray(acc_sensor.dim).item())

    #touch sensor adr
    fl_touch_sensor = model.sensor("fl_touch")
    fr_touch_sensor = model.sensor("fr_touch")
    rr_touch_sensor = model.sensor("rr_touch")
    rl_touch_sensor = model.sensor("rl_touch")

    fl_touch_sensor_adr = int(np.asarray(fl_touch_sensor.adr).item())
    fr_touch_sensor_adr = int(np.asarray(fr_touch_sensor.adr).item())
    rr_touch_sensor_adr = int(np.asarray(rr_touch_sensor.adr).item())
    rl_touch_sensor_adr = int(np.asarray(rl_touch_sensor.adr).item())

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name) # 取得 keyframe 的 ID
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data) # 計算正向動力學，更新 data.qpos 和 data.qvel
    ctrl_home = data.ctrl.copy() # 取得當前的控制輸入，作為目標控制輸入
    ctrl_range = model.actuator_ctrlrange.copy() # 取得控制輸入的範圍

    # 找 FR 的 actuator index
    fr_abd   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_hip")
    fr_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_thigh")
    fr_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_calf")

    fl_abd   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_hip")
    fl_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_thigh")
    fl_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_calf")

    rr_abd   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_hip")
    rr_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_thigh")
    rr_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_calf")

    rl_abd   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_hip")
    rl_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_thigh")
    rl_calf  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_calf")

    ctrl_state = init_walk_state()
    print(ctrl_state)
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)

            v.sync()
        cmd_node.destroy_node()
        pub_imu_node.destroy_node()
        pub_joint_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()