import numpy as np


class ComTargetPlanner:
    """
    根據要抬哪一隻腳，計算 CoM target。

    用途：
    - 輸入 swing_leg，例如 "RL"
    - 根據另外三隻支撐腳形成的三角形
    - 找出一個安全的 CoM target
    - 再根據 current_com_xy 產生 body_shift_cmd

    注意：
    這個 class 不負責求 GRF QP。
    它只負責 CoM target 與 body shift direction。
    """

    def __init__(
        self,
        max_body_shift=0.04,
        max_body_shift_step=0.001,
        k_com=0.8,
        target_mode="incenter",
        extra_bias=None,
    ):
        """
        max_body_shift:
            body shift 最大允許量。
            你之前測試 body_shift = 0.04, 0.04 有卸重效果，
            所以這裡先用 0.04 m 當上限。

        max_body_shift_step:
            每次更新 body_shift 最多變多少，避免身體突然跳動。

        k_com:
            CoM error 轉成 body shift 的比例。

        target_mode:
            "centroid"  : 三角形重心，簡單但不一定最安全
            "incenter"  : 三角形內心，離三邊距離較平均，建議用這個

        extra_bias:
            針對不同 swing leg 額外微調 target。
            預設不用。
        """

        self.legs = ["FR", "FL", "RR", "RL"]

        self.max_body_shift = float(max_body_shift)
        self.max_body_shift_step = float(max_body_shift_step)
        self.k_com = float(k_com)
        self.target_mode = target_mode

        if extra_bias is None:
            self.extra_bias = {
                "FR": np.array([0.0, 0.0]),
                "FL": np.array([0.0, 0.0]),
                "RR": np.array([0.0, 0.0]),
                "RL": np.array([0.0, 0.0]),
            }
        else:
            self.extra_bias = {
                leg: np.array(extra_bias.get(leg, [0.0, 0.0]), dtype=float)
                for leg in self.legs
            }

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def compute_target_com_xy(self, swing_leg, foot_xy_world):
        """
        根據要抬的腳，計算 target CoM xy。

        Parameters
        ----------
        swing_leg : str
            要抬起的腳，例如 "RL"

        foot_xy_world : dict
            四隻腳在 world frame 下的 xy 位置。

            形式：
            {
                "FR": np.array([x, y]),
                "FL": np.array([x, y]),
                "RR": np.array([x, y]),
                "RL": np.array([x, y]),
            }

        Returns
        -------
        dict
            {
                "target_com_xy": np.array([x, y]),
                "support_legs": [...],
                "support_points": np.array shape (3, 2),
                "centroid": np.array([x, y]),
                "incenter": np.array([x, y]),
                "is_inside": bool,
            }
        """

        self._check_swing_leg(swing_leg)
        self._check_foot_xy_world(foot_xy_world)

        support_legs = self.get_support_legs(swing_leg)

        support_points = np.array(
            [foot_xy_world[leg] for leg in support_legs],
            dtype=float,
        )

        centroid = self._triangle_centroid(support_points)
        incenter = self._triangle_incenter(support_points)

        if self.target_mode == "centroid":
            target_com_xy = centroid.copy()
        elif self.target_mode == "incenter":
            target_com_xy = incenter.copy()
        else:
            raise ValueError(
                f"Unknown target_mode: {self.target_mode}. "
                "Use 'centroid' or 'incenter'."
            )

        # 根據 swing leg 做額外微調。
        # 第一版先預設 0，之後 debug 時可以針對某隻腳補 bias。
        target_com_xy = target_com_xy + self.extra_bias[swing_leg]

        # 保險：如果 bias 把 target 推出三角形，就退回 incenter。
        is_inside = self._point_inside_triangle(target_com_xy, support_points)

        if not is_inside:
            target_com_xy = incenter.copy()
            is_inside = True

        return {
            "target_com_xy": target_com_xy,
            "support_legs": support_legs,
            "support_points": support_points,
            "centroid": centroid,
            "incenter": incenter,
            "is_inside": is_inside,
        }

    def compute_body_shift_cmd(
        self,
        current_com_xy,
        target_com_xy,
        prev_body_shift=None,
    ):
        """
        根據目前 CoM 與 target CoM，計算 body_shift_cmd。

        Parameters
        ----------
        current_com_xy : np.array([x, y])
            目前真實 CoM 的 xy。
            注意：這不是 trunk_xy。
            你現在應該用：
                current_com_xy = trunk_xy + com_offset_from_trunk

        target_com_xy : np.array([x, y])
            compute_target_com_xy() 算出的目標 CoM。

        prev_body_shift : np.array([x, y]) or None
            上一個 body_shift 指令。
            如果提供，就會做 rate limit，避免指令跳太快。

        Returns
        -------
        dict
            {
                "body_shift_cmd": np.array([x, y]),
                "raw_body_shift_cmd": np.array([x, y]),
                "com_error": np.array([x, y]),
            }
        """

        current_com_xy = np.array(current_com_xy, dtype=float)
        target_com_xy = np.array(target_com_xy, dtype=float)

        com_error = target_com_xy - current_com_xy

        raw_body_shift_cmd = self.k_com * com_error

        # 限制 body shift 最大幅度
        body_shift_cmd = self._clip_vector_norm(
            raw_body_shift_cmd,
            self.max_body_shift,
        )

        # 如果有上一個 body_shift，限制每次變化量
        if prev_body_shift is not None:
            prev_body_shift = np.array(prev_body_shift, dtype=float)

            delta = body_shift_cmd - prev_body_shift
            delta = self._clip_vector_norm(delta, self.max_body_shift_step)

            body_shift_cmd = prev_body_shift + delta

        return {
            "body_shift_cmd": body_shift_cmd,
            "raw_body_shift_cmd": raw_body_shift_cmd,
            "com_error": com_error,
        }

    def get_support_legs(self, swing_leg):
        """
        回傳支撐腳。
        例如 swing_leg = "RL"
        則 support_legs = ["FR", "FL", "RR"]
        """
        self._check_swing_leg(swing_leg)
        return [leg for leg in self.legs if leg != swing_leg]

    # ---------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------

    def _triangle_centroid(self, points):
        """
        三角形重心。
        就是三個頂點平均。
        """
        return np.mean(points, axis=0)

    def _triangle_incenter(self, points):
        """
        三角形內心。

        內心的優點：
        - 一定在三角形內
        - 到三條邊的距離比較平均
        - 比單純三角形重心更適合當安全支撐區目標
        """

        a = points[0]
        b = points[1]
        c = points[2]

        # 邊長：
        # side_a 是 b-c 的長度，對應頂點 a 的權重
        side_a = np.linalg.norm(b - c)
        side_b = np.linalg.norm(c - a)
        side_c = np.linalg.norm(a - b)

        perimeter = side_a + side_b + side_c

        if perimeter < 1e-9:
            # 三個點幾乎重疊，退回 centroid
            return self._triangle_centroid(points)

        incenter = (
            side_a * a +
            side_b * b +
            side_c * c
        ) / perimeter

        return incenter

    def _point_inside_triangle(self, point, triangle):
        """
        檢查 point 是否在 triangle 內。
        用 barycentric coordinate。
        """

        p = np.array(point, dtype=float)

        a = triangle[0]
        b = triangle[1]
        c = triangle[2]

        v0 = c - a
        v1 = b - a
        v2 = p - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01

        if abs(denom) < 1e-9:
            return False

        inv_denom = 1.0 / denom

        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        eps = 1e-9

        return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)

    def _clip_vector_norm(self, vec, max_norm):
        """
        限制向量長度。
        例如 body_shift_cmd 長度不能超過 0.04 m。
        """

        vec = np.array(vec, dtype=float)
        norm = np.linalg.norm(vec)

        if norm < 1e-9:
            return vec

        if norm > max_norm:
            return vec / norm * max_norm

        return vec

    # ---------------------------------------------------------
    # Check helpers
    # ---------------------------------------------------------

    def _check_swing_leg(self, swing_leg):
        if swing_leg not in self.legs:
            raise ValueError(
                f"swing_leg must be one of {self.legs}, got {swing_leg}"
            )

    def _check_foot_xy_world(self, foot_xy_world):
        for leg in self.legs:
            if leg not in foot_xy_world:
                raise ValueError(f"foot_xy_world missing leg: {leg}")

            arr = np.array(foot_xy_world[leg], dtype=float)

            if arr.shape != (2,):
                raise ValueError(
                    f"foot_xy_world['{leg}'] must be shape (2,), "
                    f"got {arr.shape}"
                )