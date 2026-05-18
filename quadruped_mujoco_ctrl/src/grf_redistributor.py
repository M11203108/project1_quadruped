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

def 

if __name__ == "__main__":
    for swing_leg in LEGS:
        support_legs = get_support_legs(swing_leg)
        print("swing_leg:", swing_leg, "support_legs:", support_legs)