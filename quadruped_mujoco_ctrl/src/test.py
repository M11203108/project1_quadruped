from pathlib import Path

import mujoco
import numpy as np

from robot_state import (
    build_mujoco_ids,
    read_robot_state,
)

BASE_DIR = Path(__file__).resolve().parents[2]
xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene_torque.xml"

model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)

if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
else:
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

ids = build_mujoco_ids(model)
state = read_robot_state(model, data, ids)

print("time:", state.time)
print("q shape:", state.q.shape)
print("qd shape:", state.qd.shape)
print("trunk_pos_world:", np.round(state.trunk_pos_world, 4))
print("com_body:", np.round(state.com_body, 4))
print("com_xy_body:", np.round(state.com_xy_body, 4))
print("forces:", {leg: round(f, 3) for leg, f in state.forces.items()})
print("total_force:", round(state.total_force, 3))

print("foot_xy_body:")
for leg, p in state.foot_xy_body.items():
    print(leg, np.round(p, 4))