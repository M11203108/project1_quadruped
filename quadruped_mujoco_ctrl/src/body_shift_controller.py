import numpy as np

import config as cfg
# 如果你的專案用的是 level2_control/level2_config.py，
# 則改成：
# from level2_control import level2_config as cfg


class BodyShiftController:
    """
    body_shift 平滑控制器。

    target:
        QP / planner 算出來的目標 body shift

    cmd:
        目前實際要送給 foot target builder 的 body shift

    update() 每次只讓 cmd 往 target 移動一小步。
    """

    def __init__(self):
        self.body_shift_cmd = np.array([0.0, 0.0], dtype=float)
        self.body_shift_target = np.array([0.0, 0.0], dtype=float)

    def reset(self):
        """
        把目前 body_shift_cmd 和 target 都歸零。
        通常用在：
        - 回到 HOLD
        - recovery
        - 重新開始卸重
        """

        self.body_shift_cmd = np.array([0.0, 0.0], dtype=float)
        self.body_shift_target = np.array([0.0, 0.0], dtype=float)

    def set_target(self, target):
        """
        設定 body_shift_target。

        target:
            np.array([x, y])
        """

        target = np.asarray(target, dtype=float)

        if target.shape != (2,):
            raise ValueError(f"body_shift target shape 錯誤: {target.shape}")

        self.body_shift_target = cfg.clip_body_shift(target)

    def update(self, target=None):
        """
        更新 body_shift_cmd。

        如果有傳入 target，就先設定新 target。
        然後讓 body_shift_cmd 往 body_shift_target 慢慢靠近。

        return:
            body_shift_cmd copy
        """

        if target is not None:
            self.set_target(target)

        error = self.body_shift_target - self.body_shift_cmd

        step = np.clip(
            error,
            -cfg.BODY_SHIFT_MAX_STEP,
            cfg.BODY_SHIFT_MAX_STEP,
        )

        self.body_shift_cmd = self.body_shift_cmd + step
        self.body_shift_cmd = cfg.clip_body_shift(self.body_shift_cmd)

        return self.body_shift_cmd.copy()

    def get_cmd(self):
        """
        取得目前 body_shift_cmd。
        """

        return self.body_shift_cmd.copy()

    def get_target(self):
        """
        取得目前 body_shift_target。
        """

        return self.body_shift_target.copy()

    def is_target_reached(self, tolerance=0.002):
        """
        判斷目前 body_shift_cmd 是否已經接近 target。

        tolerance:
            允許誤差，單位 m。
            0.002 代表 2 mm。
        """

        error = self.body_shift_target - self.body_shift_cmd
        err_norm = float(np.linalg.norm(error))

        return err_norm < tolerance

    def get_error(self):
        """
        回傳 target - cmd。
        """

        return self.body_shift_target - self.body_shift_cmd