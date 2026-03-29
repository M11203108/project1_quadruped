import mujoco
import numpy as np
from mujoco import viewer
import time
# from kinematics import backward_kinematics

# Load the MuJoCo model from an XML file
xml = "project1_quadruped/third_party/mujoco_menagerie/unitree_a1/scene.xml"
# A:lod model
model = mujoco.MjModel.from_xml_path(xml)
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
ctrl_des = data.ctrl.copy() # 取得當前的控制輸入，作為目標控制輸入

# 找 FR 的 actuator index
fr_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_thigh")
fr_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_calf")
fl_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_thigh")
fl_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_calf")
rr_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_thigh")
rr_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RR_calf")
rl_hip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_thigh")
rl_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "RL_calf")

# 站姿：home reset 後的 ctrl
d_hip = 0.2 # hip 抬高的幅度 rad
d_knee = -0.2 # knee 抬高的幅度 rad
d_forward = 0.12 # 前進的幅度 rad

# 抬高 FR 腿
ctrl_lift_fr = data.ctrl.copy() # 站姿的控制輸入
ctrl_lift_fr[fr_hip] = ctrl_des[fr_hip] + d_hip + d_forward # 抬高 hip
ctrl_lift_fr[fr_knee] = ctrl_des[fr_knee] + d_knee # 抬高 knee
ctrl_lift_fr[rl_hip] = ctrl_des[rl_hip] + d_hip + d_forward # 抬高 hip
ctrl_lift_fr[rl_knee] = ctrl_des[rl_knee] + d_knee # 抬高 knee

# 抬高 FL 腿
ctrl_lift_fl = data.ctrl.copy() # 站姿的控制輸入
ctrl_lift_fl[fl_hip] = ctrl_des[fl_hip] + d_hip + d_forward # 抬高 hip
ctrl_lift_fl[fl_knee] = ctrl_des[fl_knee] + d_knee # 抬高 knee
ctrl_lift_fl[rr_hip] = ctrl_des[rr_hip] + d_hip + d_forward # 抬高 hip
ctrl_lift_fl[rr_knee] = ctrl_des[rr_knee] + d_knee # 抬高 knee

t0 = time.perf_counter()
T = 1.0  # 每 1 秒切換一次

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        t = time.perf_counter() - t0
        if int(t / T) % 2 == 0:
            data.ctrl[:] = ctrl_lift_fl  # 偶數秒：站
        else:
            data.ctrl[:] = ctrl_lift_fr   # 奇數秒：抬
        mujoco.mj_step(model, data)
        v.sync()



# 使用 MuJoCo 的 viewer 來可視化模擬
# with viewer.launch_passive(model, data) as v:
#     while v.is_running():
#         data.ctrl[:] = ctrl_des
#         mujoco.mj_step(model, data)
#         v.sync()