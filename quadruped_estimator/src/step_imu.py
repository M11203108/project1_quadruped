import mujoco
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
xml = BASE_DIR / "third_party" / "mujoco_menagerie" / "unitree_a1" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

print("nsensor =", model.nsensor)

gyro_sensor = model.sensor("imu_gyro")
acc_sensor  = model.sensor("imu_acc")

gyro_adr = int(np.asarray(gyro_sensor.adr).item())
gyro_dim = int(np.asarray(gyro_sensor.dim).item())

acc_adr = int(np.asarray(acc_sensor.adr).item())
acc_dim = int(np.asarray(acc_sensor.dim).item())

while True:
    mujoco.mj_step(model, data)

    gyro = np.array(data.sensordata[gyro_adr:gyro_adr + gyro_dim], copy=True)
    acc  = np.array(data.sensordata[acc_adr:acc_adr + acc_dim], copy=True)

    print("gyro:", gyro, "acc:", acc)