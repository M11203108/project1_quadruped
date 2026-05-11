import numpy as np

ALL_LEGS = ["FR", "FL", "RR", "RL"]

FOOT_XY_BODY = {
    "FR": np.array([+0.183, -0.132]),
    "FL": np.array([+0.183, +0.132]),
    "RR": np.array([-0.183, -0.132]),
    "RL": np.array([-0.183, +0.132]),
}

def get_support_legs(swing_leg):
    support_legs = []
    for leg in ALL_LEGS:
        if leg != swing_leg:
            support_legs.append(leg)
    return support_legs

def get_support_points(support_legs):
    leg_points = []
    for leg in support_legs:
        point = FOOT_XY_BODY[leg]
        leg_points.append(point)
    return leg_points

def safety_bias(swing_leg):
    if swing_leg == "FR":
        return np.array([-0.04, 0.02])
    elif swing_leg == "FL":
        return np.array([-0.02, +0.01])
    elif swing_leg == "RR":
        return np.array([+0.02, -0.01])
    elif swing_leg == "RL":
        return np.array([+0.02, +0.01])
    else:
        return np.array([0.0, 0.0])

def point_to_segment_distance(p, a, b):
    """
    計算點 p 到線段 a-b 的最短距離。

    """
    ab = b - a
    ap = p - a
    ab_length_squared = np.dot(ab, ab)
    if ab_length_squared < 1e-9:
        return np.linalg.norm(ap)  # a 和 b 是同一點
    
    t = np.dot(ap, ab) / ab_length_squared
    t = np.clip(t, 0, 1)  # 限制 t 在 [0, 1] 範圍內
    closest = a + t * ab

    distance = np.linalg.norm(p - closest)
    return distance
  
def triangle_area(a, b, c):
    area = 0.5 * np.abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    return area

def triangle_incenter(a, b, c):
    inc = []
    len_a = np.linalg.norm(b - c)
    len_b = np.linalg.norm(a - c)
    len_c = np.linalg.norm(a - b)
    per = len_a + len_b + len_c
    inc.append((len_a * a[0] + len_b * b[0] + len_c * c[0]) / per)
    inc.append((len_a * a[1] + len_b * b[1] + len_c * c[1]) / per)
    return np.array(inc)

def point_in_triangle(p, a, b, c, eps=1e-9):
    area_abc = triangle_area(a, b, c)
    area_abp = triangle_area(a, b, p)
    area_acp = triangle_area(a, c, p)
    area_bcp = triangle_area(b, c, p)

    area_sum = area_abp + area_acp + area_bcp
    inside = np.abs(area_sum - area_abc) < eps

    return inside

def check_com_margin(swing_leg, com_xy, margin_threshold=0.02):
    """
    檢查 COM 投影是否在支撐三角形內，計算 stability margin

    margin_threshold: 最小安全距離，單位 m
    """
    support_legs = get_support_legs(swing_leg)
    support_legs_points = get_support_points(support_legs)

    a = support_legs_points[0]
    b = support_legs_points[1]
    c = support_legs_points[2]

    dis_a = point_to_segment_distance(com_xy, b, c)
    dis_b = point_to_segment_distance(com_xy, a, c)
    dis_c = point_to_segment_distance(com_xy, a, b)

    min_margin = min(dis_a, dis_b, dis_c)
    correction = triangle_incenter(a, b, c) - com_xy

    inside = point_in_triangle(com_xy, a, b, c)
    stable = inside and min_margin >= margin_threshold

    return stable, min_margin, correction
