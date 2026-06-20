import numpy as np


class TorquePDController:
    def __init__(self, Kp=60.0, Kd=2.0):
        """
        Kp:
            位置誤差

        Kd:
            速度誤差
        """
        self.Kp = Kp
        self.Kd = Kd

    def compute_pd(self, q, qd, q_des, qd_des=None):
        """
        單純 PD torque
        """

        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_des = np.asarray(q_des, dtype=float)

        if qd_des is None:
            qd_des = np.zeros_like(qd)
        else:
            qd_des = np.asarray(qd_des, dtype=float)

        tau_pd = self.Kp * (q_des - q) + self.Kd * (qd_des - qd)

        return tau_pd

    def compute(self, q, qd, q_des, qd_des=None, tau_bias=None):
        """
        PD torque + optional bias compensation。

        tau_bias:
            MuJoCo 的 qfrc_bias，包含重力 / Coriolis / bias 
            如果沒有給，就只回傳 PD torque
        """

        tau_pd = self.compute_pd(
            q=q,
            qd=qd,
            q_des=q_des,
            qd_des=qd_des,
        )

        if tau_bias is None:
            return tau_pd

        tau_bias = np.asarray(tau_bias, dtype=float)

        tau = tau_bias + tau_pd

        return tau