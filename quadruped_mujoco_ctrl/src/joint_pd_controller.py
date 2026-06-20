"""
joint_pd_controller.py

Torque PD controller.

負責：
1. 接收 IK 算出的 q_des
2. 讀目前 q, qd
3. 用 PD 算 torque
4. 加上 MuJoCo qfrc_bias 補償
5. torque clip
6. 可選擇寫入 data.ctrl

不負責：
- IK
- QP
- body_shift
- foot target
- gait phase
"""

import numpy as np

import config as cfg
# 如果你的專案是 package import，改成：
# from level2_control import level2_config as cfg


class JointPDResult:
    """
    儲存 JointPDController 的輸出結果。
    """

    def __init__(
        self,
        tau_cmd,
        tau_raw,
        tau_pd,
        tau_bias,
        q_error,
        qd_error,
        saturated,
        max_abs_tau_raw,
        max_abs_tau_cmd,
        debug,
    ):
        self.tau_cmd = tau_cmd
        self.tau_raw = tau_raw
        self.tau_pd = tau_pd
        self.tau_bias = tau_bias

        self.q_error = q_error
        self.qd_error = qd_error

        self.saturated = saturated
        self.max_abs_tau_raw = max_abs_tau_raw
        self.max_abs_tau_cmd = max_abs_tau_cmd

        self.debug = debug


class JointPDController:
    """
    關節 torque PD controller。

    輸入：
        q_des
        q
        qd
        qd_des
        tau_bias

    輸出：
        tau_cmd
    """

    def __init__(
        self,
        Kp=None,
        Kd=None,
        tau_limit=None,
        use_bias=True,
    ):
        if Kp is None:
            Kp = cfg.KP_STAND

        if Kd is None:
            Kd = cfg.KD_STAND

        if tau_limit is None:
            tau_limit = cfg.TAU_LIMIT

        self.Kp = self._make_gain_vector(Kp, "Kp")
        self.Kd = self._make_gain_vector(Kd, "Kd")

        self.tau_limit = float(tau_limit)
        self.use_bias = bool(use_bias)

        self.last_result = None

    def _make_gain_vector(self, gain, name):
        """
        gain 可以是：
        1. scalar，例如 100.0
        2. 12 維 vector，例如每個 joint 不同 Kp

        目前第一版通常用 scalar。
        """

        if np.isscalar(gain):
            return np.full(12, float(gain), dtype=float)

        gain = np.asarray(gain, dtype=float)

        if gain.shape != (12,):
            raise ValueError(
                f"{name} shape 錯誤，應該是 scalar 或 (12,), 目前是 {gain.shape}"
            )

        return gain

    def _check_vector12(self, value, name):
        """
        檢查輸入是否為 12 維 vector。
        """

        value = np.asarray(value, dtype=float)

        if value.shape != (12,):
            raise ValueError(
                f"{name} shape 錯誤，應該是 (12,), 目前是 {value.shape}"
            )

        return value

    def compute(
        self,
        q_des,
        q,
        qd,
        qd_des=None,
        tau_bias=None,
    ):
        """
        計算 torque command。

        Parameters
        ----------
        q_des:
            目標關節角，通常來自 IKController。

        q:
            目前關節角，通常來自 RobotState。

        qd:
            目前關節角速度，通常來自 RobotState。

        qd_des:
            目標關節速度。
            第一版通常用 0。

        tau_bias:
            MuJoCo qfrc_bias 對應 12 個 joint 的值。
            如果 None，則當作 0。

        Returns
        -------
        JointPDResult
        """

        q_des = self._check_vector12(q_des, "q_des")
        q = self._check_vector12(q, "q")
        qd = self._check_vector12(qd, "qd")

        if qd_des is None:
            qd_des = np.zeros(12, dtype=float)
        else:
            qd_des = self._check_vector12(qd_des, "qd_des")

        if tau_bias is None:
            tau_bias = np.zeros(12, dtype=float)
        else:
            tau_bias = self._check_vector12(tau_bias, "tau_bias")

        q_error = q_des - q
        qd_error = qd_des - qd

        tau_pd = self.Kp * q_error + self.Kd * qd_error

        if self.use_bias:
            tau_raw = tau_bias + tau_pd
        else:
            tau_raw = tau_pd.copy()

        tau_cmd = np.clip(
            tau_raw,
            -self.tau_limit,
            self.tau_limit,
        )

        saturated = bool(
            np.any(np.abs(tau_raw) > self.tau_limit + 1e-9)
        )

        max_abs_tau_raw = float(np.max(np.abs(tau_raw)))
        max_abs_tau_cmd = float(np.max(np.abs(tau_cmd)))

        debug = {
            "Kp": self.Kp.copy(),
            "Kd": self.Kd.copy(),
            "tau_limit": self.tau_limit,
            "use_bias": self.use_bias,
            "q_des": q_des.copy(),
            "q": q.copy(),
            "qd": qd.copy(),
            "qd_des": qd_des.copy(),
            "tau_bias": tau_bias.copy(),
            "tau_pd": tau_pd.copy(),
            "tau_raw": tau_raw.copy(),
            "tau_cmd": tau_cmd.copy(),
            "q_error": q_error.copy(),
            "qd_error": qd_error.copy(),
            "saturated": saturated,
            "max_abs_tau_raw": max_abs_tau_raw,
            "max_abs_tau_cmd": max_abs_tau_cmd,
        }

        result = JointPDResult(
            tau_cmd=tau_cmd,
            tau_raw=tau_raw,
            tau_pd=tau_pd,
            tau_bias=tau_bias,
            q_error=q_error,
            qd_error=qd_error,
            saturated=saturated,
            max_abs_tau_raw=max_abs_tau_raw,
            max_abs_tau_cmd=max_abs_tau_cmd,
            debug=debug,
        )

        self.last_result = result

        return result

    def get_tau_bias_from_mujoco(self, data, ids):
        """
        從 MuJoCo data.qfrc_bias 取出 12 個關節的 bias torque。

        ids 需要有：
            ids.qvel_ids

        這個 ids 就是 robot_state.py 裡 build_mujoco_ids() 回傳的物件。
        """

        tau_bias = np.array(
            [data.qfrc_bias[i] for i in ids.qvel_ids],
            dtype=float,
        )

        return tau_bias

    def compute_from_state(
        self,
        state,
        q_des,
        data=None,
        ids=None,
        qd_des=None,
    ):
        """
        從 RobotState 計算 torque。

        如果有給 data 和 ids，就會自動取 qfrc_bias。
        如果沒有，就 tau_bias = 0。

        用法：
            result = pd_controller.compute_from_state(
                state=state,
                q_des=q_des,
                data=data,
                ids=ids,
            )
        """

        if data is not None and ids is not None:
            tau_bias = self.get_tau_bias_from_mujoco(data, ids)
        else:
            tau_bias = np.zeros(12, dtype=float)

        return self.compute(
            q_des=q_des,
            q=state.q,
            qd=state.qd,
            qd_des=qd_des,
            tau_bias=tau_bias,
        )

    def write_to_mujoco(self, data, ids, tau_cmd):
        """
        將 tau_cmd 寫入 MuJoCo data.ctrl。

        ids 需要有：
            ids.actuator_ids
        """

        tau_cmd = self._check_vector12(tau_cmd, "tau_cmd")

        for actuator_id, tau in zip(ids.actuator_ids, tau_cmd):
            data.ctrl[actuator_id] = float(tau)

    def reset(self):
        """
        清除上一筆結果。
        """

        self.last_result = None