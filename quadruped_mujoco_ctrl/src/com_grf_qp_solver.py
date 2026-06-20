

import cvxpy as cp
import numpy as np

import config as cfg


class QPResult:
    """
    儲存 QP 求解結果。

    success:
        QP 是否成功求解

    status:
        CVXPY solver 狀態，例如 optimal / infeasible

    com_target_xy:
        QP 算出的 CoM 目標位置，body frame xy

    desired_forces:
        四腳 desired vertical GRF，dict 格式

    desired_force_vector:
        四腳 desired vertical GRF，vector 格式
        順序由 cfg.LEGS 決定

    slack:
        [sum_force_slack, x_moment_slack, y_moment_slack]

    objective_value:
        QP cost 數值

    debug:
        debug 用資料
    """

    def __init__(
        self,
        success,
        status,
        com_target_xy,
        desired_forces,
        desired_force_vector,
        slack,
        objective_value,
        debug,
    ):
        self.success = success
        self.status = status
        self.com_target_xy = com_target_xy
        self.desired_forces = desired_forces
        self.desired_force_vector = desired_force_vector
        self.slack = slack
        self.objective_value = objective_value
        self.debug = debug


class CoMGRFQPSolver:
    """
    CoM + vertical GRF QP solver。

    第一版設計：
        每次 solve() 都重新建立 QP。
        變數很少，所以目前不用先做 cvxpy.Parameter 優化。

    決策變數：
        fz:
            四腳垂直力 [FR, FL, RR, RL]

        com:
            CoM target xy

        slack:
            平衡式 slack
    """

    def __init__(self):
        self.last_result = None

    def _force_dict_to_vector(self, forces):
        """
        將 force dict 轉成固定順序 vector。

        cfg.LEGS = ["FR", "FL", "RR", "RL"]
        """

        return np.array(
            [float(forces[leg]) for leg in cfg.LEGS],
            dtype=float,
        )

    def _foot_xy_dict_to_matrix(self, foot_xy_body):
        """
        將 foot_xy_body dict 轉成 4x2 matrix。

        return:
            [[FR_x, FR_y],
             [FL_x, FL_y],
             [RR_x, RR_y],
             [RL_x, RL_y]]
        """

        return np.vstack([
            np.asarray(foot_xy_body[leg], dtype=float)
            for leg in cfg.LEGS
        ])

    def solve(
        self,
        swing_leg,
        foot_xy_body,
        current_com_xy_body,
        measured_forces=None,
    ):
        """
        解 CoM + vertical GRF QP。

        Parameters
        ----------
        swing_leg:
            要卸重的腳，例如 "RL"

        foot_xy_body:
            dict:
                {
                    "FR": np.array([x, y]),
                    "FL": np.array([x, y]),
                    "RR": np.array([x, y]),
                    "RL": np.array([x, y]),
                }

        current_com_xy_body:
            目前 CoM 在 trunk/body frame 的 xy

        measured_forces:
            目前實測四腳 force dict。
            若 None，使用 cfg.BASELINE_FORCE。

        Returns
        -------
        QPResult
        """

        if swing_leg not in cfg.LEGS:
            raise ValueError(f"未知 swing_leg: {swing_leg}")

        support_legs = cfg.get_support_legs(swing_leg)

        foot_xy = self._foot_xy_dict_to_matrix(foot_xy_body)
        foot_x = foot_xy[:, 0]
        foot_y = foot_xy[:, 1]

        current_com_xy_body = np.asarray(
            current_com_xy_body,
            dtype=float,
        )

        if current_com_xy_body.shape != (2,):
            raise ValueError(
                f"current_com_xy_body shape 錯誤: {current_com_xy_body.shape}"
            )

        if measured_forces is None:
            measured_force_vec = cfg.get_baseline_force_vector()
        else:
            measured_force_vec = self._force_dict_to_vector(measured_forces)

        swing_idx = cfg.LEG_INDEX[swing_leg]

        support_mask = np.zeros(4)
        for leg in support_legs:
            support_mask[cfg.LEG_INDEX[leg]] = 1.0

        swing_mask = np.zeros(4)
        swing_mask[swing_idx] = 1.0

        W = float(cfg.TOTAL_WEIGHT)

        # ============================================================
        # Decision variables
        # ============================================================

        fz = cp.Variable(4)
        com = cp.Variable(2)
        slack = cp.Variable(3)

        # ============================================================
        # Constraints
        # ============================================================

        constraints = []

        for leg in cfg.LEGS:
            idx = cfg.LEG_INDEX[leg]

            if leg == swing_leg:
                constraints += [
                    fz[idx] >= cfg.SWING_FZ_MIN,
                    fz[idx] <= cfg.SWING_FZ_MAX,
                ]
            else:
                constraints += [
                    fz[idx] >= cfg.SUPPORT_FZ_MIN,
                    fz[idx] <= cfg.FZ_MAX,
                ]

        # 垂直力總和：
        # sum(Fz) ≈ mg
        constraints += [
            cp.sum(fz) + slack[0] == W,
        ]

        # CoM / GRF 平衡：
        # sum(x_i * F_i) ≈ W * com_x
        # sum(y_i * F_i) ≈ W * com_y
        constraints += [
            foot_x @ fz + slack[1] == W * com[0],
            foot_y @ fz + slack[2] == W * com[1],
        ]

        # CoM target 不要離目前 CoM 太遠
        constraints += [
            com[0] >= current_com_xy_body[0] - cfg.COM_TARGET_LIMIT_X,
            com[0] <= current_com_xy_body[0] + cfg.COM_TARGET_LIMIT_X,
            com[1] >= current_com_xy_body[1] - cfg.COM_TARGET_LIMIT_Y,
            com[1] <= current_com_xy_body[1] + cfg.COM_TARGET_LIMIT_Y,
        ]

        # ============================================================
        # Force reference
        # ============================================================

        force_ref = measured_force_vec.copy()

        # swing leg 希望卸到目標值，第一版通常是 0 N
        force_ref[swing_idx] = cfg.SWING_FORCE_TARGET

        # support legs 平均分擔剩餘重量
        remaining_weight = W - cfg.SWING_FORCE_TARGET
        support_force = remaining_weight / len(support_legs)

        for leg in support_legs:
            idx = cfg.LEG_INDEX[leg]
            force_ref[idx] = support_force

        # ============================================================
        # Cost function
        # ============================================================

        swing_force = fz[swing_idx]

        support_force_error = cp.multiply(
            support_mask,
            fz - force_ref,
        )

        force_tracking_cost = cp.sum_squares(
            fz - force_ref
        )

        support_regularization_cost = cp.sum_squares(
            support_force_error
        )

        swing_unload_cost = cp.square(
            swing_force - cfg.SWING_FORCE_TARGET
        )

        com_shift_cost = cp.sum_squares(
            com - current_com_xy_body
        )

        slack_cost = (
            cfg.QP_WEIGHT_SLACK_SUM_FORCE * cp.square(slack[0])
            +
            cfg.QP_WEIGHT_SLACK_MOMENT * cp.sum_squares(slack[1:3])
        )

        objective = cp.Minimize(
            cfg.QP_WEIGHT_FORCE_TRACKING * force_tracking_cost
            +
            cfg.QP_WEIGHT_FORCE_REGULARIZATION * support_regularization_cost
            +
            cfg.QP_WEIGHT_SWING_UNLOAD * swing_unload_cost
            +
            cfg.QP_WEIGHT_COM_SHIFT * com_shift_cost
            +
            slack_cost
        )

        # ============================================================
        # Solve
        # ============================================================

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=100000,
                eps_abs=1e-5,
                eps_rel=1e-5,
                polish=True,
            )
        except Exception as exc:
            return self._make_failed_result(
                status=f"exception: {exc}",
                foot_xy=foot_xy,
                current_com_xy=current_com_xy_body,
                measured_force_vec=measured_force_vec,
                force_ref=force_ref,
            )

        success = problem.status in [
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        ]

        if not success:
            return self._make_failed_result(
                status=problem.status,
                foot_xy=foot_xy,
                current_com_xy=current_com_xy_body,
                measured_force_vec=measured_force_vec,
                force_ref=force_ref,
            )

        # ============================================================
        # Extract result
        # ============================================================

        fz_value = np.asarray(fz.value, dtype=float)
        com_value = np.asarray(com.value, dtype=float)
        slack_value = np.asarray(slack.value, dtype=float)

        desired_forces = {
            leg: float(fz_value[cfg.LEG_INDEX[leg]])
            for leg in cfg.LEGS
        }

        debug = {
            "swing_leg": swing_leg,
            "support_legs": support_legs,
            "foot_xy": foot_xy,
            "current_com_xy": current_com_xy_body,
            "measured_force_vec": measured_force_vec,
            "force_ref": force_ref,
            "support_mask": support_mask,
            "swing_mask": swing_mask,
            "total_weight": W,
            "solver_status": problem.status,
        }

        result = QPResult(
            success=True,
            status=problem.status,
            com_target_xy=com_value,
            desired_forces=desired_forces,
            desired_force_vector=fz_value,
            slack=slack_value,
            objective_value=float(problem.value),
            debug=debug,
        )

        self.last_result = result

        return result

    def _make_failed_result(
        self,
        status,
        foot_xy,
        current_com_xy,
        measured_force_vec,
        force_ref,
    ):
        """
        QP 失敗時回傳安全結果。

        注意：
        外部程式如果看到 success=False，
        就不應該使用 com_target_xy 或 desired_forces。
        """

        debug = {
            "foot_xy": foot_xy,
            "current_com_xy": current_com_xy,
            "measured_force_vec": measured_force_vec,
            "force_ref": force_ref,
            "solver_status": status,
        }

        desired_forces = {
            leg: float("nan")
            for leg in cfg.LEGS
        }

        result = QPResult(
            success=False,
            status=status,
            com_target_xy=np.array([np.nan, np.nan], dtype=float),
            desired_forces=desired_forces,
            desired_force_vector=np.full(4, np.nan),
            slack=np.full(3, np.nan),
            objective_value=np.nan,
            debug=debug,
        )

        self.last_result = result

        return result