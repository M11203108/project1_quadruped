import mujoco
import numpy as np
from mujoco import viewer
import time
from kinematics import backward_kinematics_3d, backward_kinematics_2d

# Load the MuJoCo model from an XML file
xml = "project1_quadruped/third_party/mujoco_menagerie/unitree_a1/scene.xml"
# A:lod model
model = mujoco.MjModel.from_xml_path(xml)

# h = 0.08505
hu = 0.2
hl = 0.2

x0, y0, z0 = 0.0, 0.0, -0.25

# print(backward_kinematics_3d(x0, y0, z0, h, hu, hl))
# print(np.rad2deg(backward_kinematics_3d(x0, y0, z0, h, hu, hl)))
# print(backward_kinematics_3d(x0 + 0.02, y0, z0, h, hu, hl))
# print(backward_kinematics_3d(x0 + 0.05, y0, z0, h, hu, hl))

print(backward_kinematics_2d(0.0, -0.25, hu, hl))
print(np.rad2deg(backward_kinematics_2d(0.0, -0.25, hu, hl)))