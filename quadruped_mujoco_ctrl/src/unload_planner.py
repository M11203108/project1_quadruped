"""
unload_planner.py

Level 2 卸重策略層。

負責：
1. 呼叫 CoM + vertical GRF QP solver
2. 把 com_target 轉成 body_shift_target
3. 根據實測 force 判斷是否卸重成功
4. 檢查 support legs 是否安全
5. 必要時加一點 force feedback correction

不負責：
- body_shift 平滑
- IK
- torque control
- MuJoCo sensor 讀取
- gait 順序決定
"""

import numpy as np

import config as cfg
from com_grf_qp_solver import CoMGRFQPSolver

# 如果你的專案是 level2_control package，則改成：
# from level2_control import level2_config as cfg
# from level2_control.com_grf_qp_solver import CoMGRFQPSolver


class UnloadPlanResult:
    """
    儲存 UnloadPlanner 的結果。

    main 之後只需要看這個物件，不要直接拆 QP solver。
    """

    def __init__(
        self,
        success,
        reason,
        swing_leg,
        support_legs,
        body_shift_target,
        qp_result,
        desired_forces,
        force_error,
        swing_force,
        swing_force_ratio,
        unload_success,
        support_ok,
        feedback_correction,
        debug,
    ):
        self.success = success
        self.reason = reason

        self.swing_leg = swing_leg
        self.support_legs = support_legs

        self.body_shift_target = body_shift_target

        self.qp_result = qp_result
        self.desired_forces = desired_forces
        self.force_error = force_error

        self.swing_force = swing_force
        self.swing_force_ratio = swing_force_ratio

        self.unload_success = unload_success
        self.support_ok = support_ok

        self.feedback_correction = feedback_correction

        self.debug = debug


class UnloadPlanner:
    """
    卸重規劃器。

    用法：
        planner = UnloadPlanner()

        result = planner.plan(
            state=state,
            swing_leg="RL",
        )

        body_shift_target = result.body_shift_target
    """

    def __init__(self, qp_solver=None, enable_force_feedback=False):
        if qp_solver is None:
            self.qp_solver = CoMGRFQPSolver()
        else:
            self.qp_solver = qp_solver

        self.enable_force_feedback = enable_force_feedback

        self.last_body_shift_target = np.array([0.0, 0.0], dtype=float)
        self.last_result = None

    def reset(self):
        """
        重置 planner 狀態。
        """

        self.last_body_shift_target = np.array([0.0, 0.0], dtype=float)
        self.last_result = None

    def plan(self, state, swing_leg):
        """
        根據目前 RobotState 和 swing_leg 規劃卸重。

        Parameters
        ----------
        state:
            RobotState。
            需要有：
                state.com_xy_body
                state.foot_xy_body
                state.forces

        swing_leg:
            要卸重的腳，例如 "RL"

        Returns
        -------
        UnloadPlanResult
        """

        if swing_leg not in cfg.LEGS:
            raise ValueError(f"未知 swing_leg: {swing_leg}")

        support_legs = cfg.get_support_legs(swing_leg)

        # ============================================================
        # 1. 呼叫 QP solver
        # ============================================================

        qp_result = self.qp_solver.solve(
            swing_leg=swing_leg,
            foot_xy_body=state.foot_xy_body,
            current_com_xy_body=state.com_xy_body,
            measured_forces=state.forces,
        )

        if not qp_result.success:
            return self._make_failed_result(
                reason=f"QP failed: {qp_result.status}",
                state=state,
                swing_leg=swing_leg,
                support_legs=support_legs,
                qp_result=qp_result,
            )

        # ============================================================
        # 2. com_target 轉成 body_shift_target
        # ============================================================

        # QP 算的是「希望 CoM 到哪裡」
        # body_shift_target 是「身體要移多少」
        body_shift_target = (
            qp_result.com_target_xy - state.com_xy_body
        )

        body_shift_target = cfg.clip_body_shift(body_shift_target)

        # ============================================================
        # 3. 計算 desired force 和 measured force 的誤差
        # ============================================================

        desired_forces = qp_result.desired_forces

        force_error = {}
        for leg in cfg.LEGS:
            desired = float(desired_forces[leg])
            measured = float(state.forces[leg])
            force_error[leg] = desired - measured

        # ============================================================
        # 4. 判斷 swing leg 是否卸重成功
        # ============================================================

        swing_force = float(state.forces[swing_leg])
        baseline_swing_force = float(cfg.BASELINE_FORCE[swing_leg])

        if baseline_swing_force > 1e-6:
            swing_force_ratio = swing_force / baseline_swing_force
        else:
            swing_force_ratio = 999.0

        abs_ok = swing_force <= cfg.UNLOAD_FORCE_ABS_THRESHOLD
        ratio_ok = swing_force_ratio <= cfg.UNLOAD_FORCE_RATIO_THRESHOLD

        # ============================================================
        # 5. 檢查 support legs 是否安全
        # ============================================================

        support_ok = True
        support_reasons = []

        for leg in support_legs:
            force = float(state.forces[leg])

            if force < cfg.MIN_SUPPORT_FORCE:
                support_ok = False
                support_reasons.append(
                    f"{leg} force too low: {force:.2f}"
                )

            if force > cfg.MAX_SUPPORT_FORCE:
                support_ok = False
                support_reasons.append(
                    f"{leg} force too high: {force:.2f}"
                )

        unload_success = bool((abs_ok or ratio_ok) and support_ok)

        # ============================================================
        # 6. force feedback correction
        # ============================================================

        feedback_correction = np.array([0.0, 0.0], dtype=float)

        if self.enable_force_feedback and not unload_success:
            feedback_correction = self._compute_force_feedback_correction(
                body_shift_target=body_shift_target,
                swing_force=swing_force,
                baseline_swing_force=baseline_swing_force,
            )

            body_shift_target = body_shift_target + feedback_correction
            body_shift_target = cfg.clip_body_shift(body_shift_target)

        # ============================================================
        # 7. 整理 reason / debug
        # ============================================================

        if unload_success:
            reason = "unload_success"
        elif not support_ok:
            reason = "support_not_ok: " + ", ".join(support_reasons)
        elif not abs_ok and not ratio_ok:
            reason = (
                f"swing_force_not_low_enough: "
                f"{swing_force:.2f} N, ratio={swing_force_ratio:.2f}"
            )
        else:
            reason = "unload_not_ready"

        debug = {
            "qp_status": qp_result.status,
            "com_xy_body": state.com_xy_body.copy(),
            "com_target_xy": qp_result.com_target_xy.copy(),
            "raw_body_shift_target": (
                qp_result.com_target_xy - state.com_xy_body
            ),
            "body_shift_target": body_shift_target.copy(),
            "desired_forces": desired_forces,
            "measured_forces": state.forces,
            "force_error": force_error,
            "swing_force": swing_force,
            "swing_force_ratio": swing_force_ratio,
            "abs_ok": abs_ok,
            "ratio_ok": ratio_ok,
            "support_ok": support_ok,
            "support_reasons": support_reasons,
            "feedback_correction": feedback_correction.copy(),
        }

        result = UnloadPlanResult(
            success=True,
            reason=reason,
            swing_leg=swing_leg,
            support_legs=support_legs,
            body_shift_target=body_shift_target,
            qp_result=qp_result,
            desired_forces=desired_forces,
            force_error=force_error,
            swing_force=swing_force,
            swing_force_ratio=swing_force_ratio,
            unload_success=unload_success,
            support_ok=support_ok,
            feedback_correction=feedback_correction,
            debug=debug,
        )

        self.last_body_shift_target = body_shift_target.copy()
        self.last_result = result

        return result

    def _compute_force_feedback_correction(
        self,
        body_shift_target,
        swing_force,
        baseline_swing_force,
    ):
        """
        根據 swing leg 目前 force，沿著 body_shift_target 方向補一點修正。

        注意：
        這不是主要控制，只是補償。
        主要方向仍由 QP 決定。
        """

        shift_norm = float(np.linalg.norm(body_shift_target))

        if shift_norm < cfg.UNLOAD_SHIFT_EPS:
            return np.array([0.0, 0.0], dtype=float)

        direction = body_shift_target / shift_norm

        # 超過 threshold 的比例
        threshold = cfg.UNLOAD_FORCE_ABS_THRESHOLD
        excess_force = max(0.0, swing_force - threshold)

        if baseline_swing_force > 1e-6:
            excess_ratio = excess_force / baseline_swing_force
        else:
            excess_ratio = 0.0

        correction_mag = cfg.UNLOAD_FEEDBACK_GAIN * excess_ratio

        correction_mag = float(
            np.clip(
                correction_mag,
                0.0,
                cfg.UNLOAD_FEEDBACK_MAX_CORRECTION,
            )
        )

        correction = direction * correction_mag

        return correction

    def _make_failed_result(
        self,
        reason,
        state,
        swing_leg,
        support_legs,
        qp_result,
    ):
        """
        QP 失敗時回傳安全結果。
        """

        body_shift_target = self.last_body_shift_target.copy()

        desired_forces = {
            leg: float("nan")
            for leg in cfg.LEGS
        }

        force_error = {
            leg: float("nan")
            for leg in cfg.LEGS
        }

        swing_force = float(state.forces[swing_leg])
        baseline_swing_force = float(cfg.BASELINE_FORCE[swing_leg])

        if baseline_swing_force > 1e-6:
            swing_force_ratio = swing_force / baseline_swing_force
        else:
            swing_force_ratio = 999.0

        debug = {
            "qp_status": qp_result.status,
            "com_xy_body": state.com_xy_body.copy(),
            "body_shift_target": body_shift_target.copy(),
            "measured_forces": state.forces,
            "swing_force": swing_force,
            "swing_force_ratio": swing_force_ratio,
        }

        result = UnloadPlanResult(
            success=False,
            reason=reason,
            swing_leg=swing_leg,
            support_legs=support_legs,
            body_shift_target=body_shift_target,
            qp_result=qp_result,
            desired_forces=desired_forces,
            force_error=force_error,
            swing_force=swing_force,
            swing_force_ratio=swing_force_ratio,
            unload_success=False,
            support_ok=False,
            feedback_correction=np.array([0.0, 0.0], dtype=float),
            debug=debug,
        )

        self.last_result = result

        return result