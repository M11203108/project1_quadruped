"""
foot_target_builder.py

把 body_shift_cmd / lift_height 轉成四隻腳的 foot_targets_body。

這個檔案只做：
1. 從 nominal foot position 建立 foot target
2. 對 support legs 套用 body_shift
3. 對 swing leg 套用 lift_height

不做：
- QP
- force feedback
- body_shift 平滑
- IK
- torque control
- MuJoCo sensor 讀取
"""

import numpy as np

import config as cfg
# 如果你使用 package import，改成：
# from level2_control import level2_config as cfg


class FootTargetBuilder:
    """
    foot target 建立器。

    輸入：
        body_shift_cmd
        swing_leg
        lift_height

    輸出：
        foot_targets_body dict
    """

    def __init__(self):
        pass

    def build(
        self,
        body_shift_cmd,
        swing_leg=None,
        lift_height=0.0,
        apply_shift_to_swing=False,
    ):
        """
        建立四隻腳在 body frame 下的 foot targets。

        Parameters
        ----------
        body_shift_cmd:
            np.array([x, y])
            身體希望移動的 xy

        swing_leg:
            目前要卸重或抬起的腳。
            可以是 "FR" / "FL" / "RR" / "RL" / None

        lift_height:
            swing leg 抬腳高度，單位 m。
            卸重階段通常是 0。
            抬腳階段可以是 0.02 ~ 0.04。

        apply_shift_to_swing:
            False:
                只對 support legs 套 body_shift。
                這是目前建議的第一版。

            True:
                四隻腳都套 body_shift。
                之後如果你想測可以打開。

        Returns
        -------
        foot_targets_body:
            {
                "FR": np.array([x, y, z]),
                ...
            }
        """

        body_shift_cmd = np.asarray(body_shift_cmd, dtype=float)

        if body_shift_cmd.shape != (2,):
            raise ValueError(
                f"body_shift_cmd shape 錯誤: {body_shift_cmd.shape}"
            )

        body_shift_cmd = cfg.clip_body_shift(body_shift_cmd)

        if swing_leg is not None and swing_leg not in cfg.LEGS:
            raise ValueError(f"未知 swing_leg: {swing_leg}")

        if swing_leg is None:
            support_legs = cfg.LEGS.copy()
        else:
            support_legs = cfg.get_support_legs(swing_leg)

        foot_targets_body = {}

        for leg in cfg.LEGS:
            # 從 nominal foot xyz 複製一份
            target = cfg.NOMINAL_FOOT_XYZ_BODY[leg].copy()

            # body 要往 +x/+y 移，foot target 要往 -x/-y 移
            should_apply_shift = (
                leg in support_legs
                or apply_shift_to_swing
                or swing_leg is None
            )

            if should_apply_shift:
                target[0] -= body_shift_cmd[0]
                target[1] -= body_shift_cmd[1]

            # swing leg 抬腳
            # 注意：你的 nominal z 是負值。
            # 抬腳代表 foot target z 往上，數值增加。
            if leg == swing_leg:
                target[2] += float(lift_height)

            foot_targets_body[leg] = target

        return foot_targets_body

    def build_standing(self):
        """
        回傳 nominal standing foot targets。
        """

        return {
            leg: cfg.NOMINAL_FOOT_XYZ_BODY[leg].copy()
            for leg in cfg.LEGS
        }