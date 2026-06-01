import numpy as np

from kinematics import backward_kinematics_3d
from robot_interface import LEGS, LEG_SIDE_SIGN

class IKController:
    def __init__(self, h=0.08505, hu=0.2, hl=0.2):
        """
        h:
            hip lateral offset，髖關節左右偏移距離
        hu:
            upper leg / thigh length，大腿長度
        hl:
            lower leg / calf length，小腿長度
        """
        self.h = h
        self.hu = hu
        self.hl = hl

    def solve_leg(self, leg, foot_target_body, hip_body):
        """
        單腳 IK
        """

        foot_target_body = np.asarray(foot_target_body, dtype=float)
        hip_body = np.asarray(hip_body, dtype=float)

        # body frame → 單腳 hip frame
        foot_target_leg = foot_target_body - hip_body

        side_angle = LEG_SIDE_SIGN[leg]

        abd, hip, knee = backward_kinematics_3d(
            foot_target_leg[0],
            foot_target_leg[1],
            foot_target_leg[2],
            self.h,
            self.hu,
            self.hl,
            side_angle=side_angle,
        )

        q_leg = np.array([abd, hip, knee], dtype=float)

        return q_leg

    def solve(self, foot_targets_body, hip_xyz_body):
        """
        四隻腳 IK
        """

        q_des_list = []

        for leg in LEGS:
            q_leg = self.solve_leg(
                leg=leg,
                foot_target_body=foot_targets_body[leg],
                hip_body=hip_xyz_body[leg],
            )

            q_des_list.extend(q_leg)

        q_des = np.array(q_des_list, dtype=float)

        if q_des.shape != (12,):
            raise RuntimeError(f"q_des shape 錯誤，目前是 {q_des.shape}")

        if not np.all(np.isfinite(q_des)):
            raise RuntimeError(f"q_des 出現 NaN 或 inf: {q_des}")

        return q_des