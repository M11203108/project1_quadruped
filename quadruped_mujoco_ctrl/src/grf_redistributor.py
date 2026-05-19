import numpy as np

LEGS = ["FR", "FL", "RR", "RL"]

FOOT_XY_BODY = {
    "FR": np.array([+0.183, -0.132]),
    "FL": np.array([+0.183, +0.132]),
    "RR": np.array([-0.183, -0.132]),
    "RL": np.array([-0.183, +0.132]),
}

def get_support_legs(swing_leg):
    support_legs = []
    for leg in LEGS:
        if leg != swing_leg:
            support_legs.append(leg)
    return support_legs

def compute_cop_xy(forces):
    """
    輸入:每支腳的回傳力
    輸出:壓力中心位置
    """
    total_force = 0.0
    weighted_sum = np.zeros(2)
    for leg in LEGS:
        force = max(float(forces.get(leg, 0.0)), 0.0)  #每腳的回傳力 確保是正的
        total_force += force
        weighted_sum += force*FOOT_XY_BODY[leg]
    
    if total_force < 1e-6:
        return np.array([0.0, 0.0])  #避免除以零
    return weighted_sum / total_force

def compute_target_cop_xy(swing_leg, unload_gain=0.15):
    """
    輸入: 擺動腳
    輸出: 目標壓力中心位置
    """
    support_legs = get_support_legs(swing_leg)

    support_points = []
    for leg in support_legs:
        support_points.append(FOOT_XY_BODY[leg])
    support_points = np.array(support_points)
    support_center = np.mean(support_points, axis=0) #mean 矩陣平均
    swing_point = FOOT_XY_BODY[swing_leg]
    away_vec = support_center - swing_point
    target_cop = support_center + unload_gain*away_vec
    return target_cop

def solve_desired_grf(swing_leg, forces):
    """
    卸重支撐腳的支撐力
    """
    support_legs = get_support_legs(swing_leg)
    W = 0.0
    for leg in LEGS:
        W += max(float(forces.get(leg, 0.0)), 0.0)

    target_cop = compute_target_cop_xy(swing_leg)
    support_points = []
    for leg in support_legs:
        support_points.append(FOOT_XY_BODY[leg])
    support_points = np.array(support_points)
    A = np.array([
        [1.0, 1.0, 1.0],
        [support_points[0][0], support_points[1][0], support_points[2][0]],
        [support_points[0][1], support_points[1][1], support_points[2][1]],
    ])

    b = np.array([
        W,
        W * target_cop[0],
        W * target_cop[1],
    ])

    f_support = np.linalg.solve(A, b)
    f_min = 5.0
    f_max = 70.0

    f_support = np.clip(f_support, f_min, f_max)

    total_after_clip = np.sum(f_support)
    if total_after_clip > 1e-6:
        f_support = f_support * (W / total_after_clip)

    desired_forces = {leg: 0.0 for leg in LEGS} #4隻腳初始為0
    for leg ,force in zip(support_legs, f_support): #zip 兩個 list配起來
        desired_forces[leg] = float(force)

    desired_forces[swing_leg] = 0.0

    measured_cop = compute_cop_xy(forces)
    debug_info = {
        "support_legs": support_legs,
        "W": W,
        "measured_cop": measured_cop,
        "target_cop": target_cop,
        "f_support": f_support,
    }
    return desired_forces, debug_info

def compute_force_error(desired_forces, forces):
    """
    輸入: desired期望支撐力，measured每腳的回傳力
    輸出: 每腳的力誤差
    """
    force_error = {}

    for leg in LEGS:
        desired = float(desired_forces.get(leg, 0.0)) #0.0=沒找到就是0.0
        measured = float(forces.get(leg, 0.0))
        force_error[leg] = desired - measured

    return force_error

def compute_cop_error(swing_leg, forces):
    target = compute_target_cop_xy(swing_leg)
    measured = compute_cop_xy(forces)
    cop_error = target - measured
    return cop_error

def compute_grf_redistribution(swing_leg, forces):
    desired_forces, debug_info = solve_desired_grf(swing_leg, forces)
    force_error = compute_force_error(desired_forces, forces)
    cop_error = compute_cop_error(swing_leg, forces)

    debug_info["force_error"] = force_error
    debug_info["cop_error"] = cop_error

    return desired_forces, force_error, cop_error, debug_info

if __name__ == "__main__":
    forces = {
        "FR": 10.0,
        "FL": 10.0,
        "RR": 10.0,
        "RL": 10.0,
    }
    for swing_leg in LEGS:
        desired_forces, force_error, cop_error, debug = compute_grf_redistribution(
            swing_leg,
            forces
        )

        print("\n====================")
        print("swing_leg:", swing_leg)
        print("desired_forces:", desired_forces)
        print("force_error:", force_error)
        print("measured_cop:", debug["measured_cop"])
        print("target_cop:", debug["target_cop"])
        print("cop_error:", cop_error)
