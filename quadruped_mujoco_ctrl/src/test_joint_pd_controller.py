import numpy as np

from joint_pd_controller import JointPDController

controller = JointPDController()

q = np.zeros(12)
qd = np.zeros(12)

q_des = np.ones(12) * 0.1
tau_bias = np.ones(12) * 2.0

q_des_big = np.ones(12) * 1.0

result_big = controller.compute(
    q_des=q_des_big,
    q=q,
    qd=qd,
    tau_bias=tau_bias,
)

print("\nBig command test")
print("tau_raw:", np.round(result_big.tau_raw, 3))
print("tau_cmd:", np.round(result_big.tau_cmd, 3))
print("saturated:", result_big.saturated)
print("max_abs_tau_raw:", round(result_big.max_abs_tau_raw, 3))
print("max_abs_tau_cmd:", round(result_big.max_abs_tau_cmd, 3))