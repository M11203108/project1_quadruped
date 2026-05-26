import mujoco
import numpy as np
from mujoco import viewer
import time
from pathlib import Path
# from kinematics import backward_kinematics

# Load the MuJoCo model from an XML file
BASE_DIR = Path(__file__).resolve().parents[2]
xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"
# A:lod model
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
print("joint數 model.njnt =", model.njnt)
print("actuator數 model.nu =", model.nu)

print("\n=== Actuators ===")
for i in range(model.nu):
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(i, act_name)

key_name = "home"
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name) # 取得 keyframe 的 ID
mujoco.mj_resetDataKeyframe(model, data, key_id)

mujoco.mj_forward(model, data) # 計算正向動力學，更新 data.qpos 和 data.qvel
q_des = data.qpos[7:19].copy()
qd_des = np.zeros(12)

Kp = 60.0
Kd = 2.0
tau_limit = 33.5

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_forward(model, data)

        q = data.qpos[7:19].copy()
        qd = data.qvel[6:18].copy()

        tau_pd = Kp * (q_des - q) + Kd * (qd_des - qd)

        # joint gravity / bias compensation
        tau_ff = data.qfrc_bias[6:18].copy()

        tau = tau_ff + tau_pd
        tau = np.clip(tau, -tau_limit, tau_limit)
        print(
            "z:", round(data.qpos[2], 3),
            "max_err:", round(np.max(np.abs(q_des - q)), 3),
            "max_tau:", round(np.max(np.abs(tau)), 2),
        )

        data.ctrl[:] = tau

        mujoco.mj_step(model, data)
        v.sync()