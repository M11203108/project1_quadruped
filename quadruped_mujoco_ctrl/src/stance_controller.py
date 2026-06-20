import mujoco
import numpy as np

from project1_quadruped.src.quadruped_mujoco_ctrl.src.config import LEGS


class StanceController:
    def __init__(
        self,
        force_gain=0.6,
        max_force_cmd=12.0,
        max_tau_grf=3.0,
        jtf_sign=-1.0,
    ):
        """
        force_gain:
            force error 的比例增益。
            fz_cmd = force_gain * (desired_fz - measured_fz)

        max_force_cmd:
            每隻腳最大修正力，避免一開始太暴力。

        max_tau_grf:
            每個關節由 GRF 產生的最大 torque。

        jtf_sign:
            JᵀF 的符號。
            如果測試後發現力變化方向相反，就改成 -1.0。
        """
        self.force_gain = force_gain
        self.max_force_cmd = max_force_cmd
        self.max_tau_grf = max_tau_grf
        self.jtf_sign = jtf_sign

    def get_leg_slice(self, leg):
        """
        回傳某隻腳在 12 維 tau 裡的位置。
        LEGS = ["FR", "FL", "RR", "RL"]
        每隻腳 3 個關節。
        """
        leg_id = LEGS.index(leg)
        start = leg_id * 3
        end = start + 3

        return start, end

    def compute_leg_jacobian(self, model, data, ids, leg):
        """
        計算某隻腳 foot site 的 translational Jacobian。

        MuJoCo 的 mj_jacSite 給的是 world frame Jacobian。
        J_leg shape = (3, 3)
        """
        site_id = ids["site"][leg]

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        start, end = self.get_leg_slice(leg)

        qvel_cols = ids["qvel"][start:end]

        J_leg = jacp[:, qvel_cols]

        return J_leg

    def compute(
        self,
        model,
        data,
        ids,
        state,
        desired_forces,
        controlled_legs,
    ):
        """
        input:
            desired_forces:
                QP 算出來的四腳 desired Fz

            state.forces:
                MuJoCo touch sensor 量到的四腳 measured force

            support_legs:
                目前支撐腳，例如 ["FR", "FL", "RR"]

        output:
            tau_grf:
                12 維 GRF correction torque

            debug:
                每隻腳的 force error / fz_cmd / tau_leg
        """

        tau_grf = np.zeros(12)

        debug = {
            "per_leg": {},
            "max_tau_grf": 0.0,
        }

        for leg in LEGS:
            start, end = self.get_leg_slice(leg)

            # swing leg 第一版不加 GRF torque
            if leg not in controlled_legs:
                debug["per_leg"][leg] = {
                    "is_support": False,
                    "desired_fz": float(desired_forces.get(leg, 0.0)),
                    "measured_fz": float(state.forces[leg]),
                    "fz_error": 0.0,
                    "fz_cmd": 0.0,
                    "tau_leg": np.zeros(3),
                }
                continue

            desired_fz = float(desired_forces[leg])
            measured_fz = float(state.forces[leg])

            fz_error = desired_fz - measured_fz

            fz_cmd = self.force_gain * fz_error
            fz_cmd = np.clip(
                fz_cmd,
                -self.max_force_cmd,
                self.max_force_cmd,
            )

            # body frame 下的垂直力修正
            F_body = np.array([
                0.0,
                0.0,
                fz_cmd,
            ])

            # mj_jacSite 是 world frame Jacobian，所以 force 也要轉到 world frame
            F_world = np.array([
                    0.0,
                    0.0,
                    fz_cmd,
                ])

            J_leg = self.compute_leg_jacobian(
                model=model,
                data=data,
                ids=ids,
                leg=leg,
            )

            tau_leg = self.jtf_sign * (J_leg.T @ F_world)

            tau_leg = np.clip(
                tau_leg,
                -self.max_tau_grf,
                self.max_tau_grf,
            )

            tau_grf[start:end] = tau_leg

            debug["per_leg"][leg] = {
                "is_support": True,
                "desired_fz": desired_fz,
                "measured_fz": measured_fz,
                "fz_error": fz_error,
                "fz_cmd": float(fz_cmd),
                "F_body": F_body,
                "F_world": F_world,
                "J_leg": J_leg,
                "tau_leg": tau_leg,
            }

        debug["max_tau_grf"] = float(np.max(np.abs(tau_grf)))

        return tau_grf, debug