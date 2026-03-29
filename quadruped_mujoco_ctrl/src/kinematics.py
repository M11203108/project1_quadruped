import numpy as np

def backward_kinematics_3d(x, y, z, h, hu, hl):
    """
    計算四足機器人腿部的關節角度。
    
    Parameters:
    x: 末端位置的x座標
    y: 末端位置的y座標
    z: 末端位置的z座標
    h: 髖部長度
    hu: 大腿長度
    hl: 小腿長度
    
    Returns:
    abduction_angle: 外展關節角度 (rad)
    hip_angle: 髖關節角度 (rad)
    knee_angle: 膝關節角度 (rad)
    """
    D = np.sqrt(np.square(y) + np.square(z))
    L = np.sqrt(np.square(D) - np.square(h))

    # 計算膝關節角度
    S = np.sqrt(np.square(x) + np.square(z))
    N = (np.square(S) - np.square(hl) - np.square(hu)) / (2 * hu * hl)
    N = np.clip(N, -1.0, 1.0)  # 確保 n 在 [-1, 1] 範圍
    knee_angle = -np.arccos(N)

    # 計算髖關節角度
    hip_angle = np.arctan2(x, L) - np.arctan2(hl * np.sin(knee_angle), hu + hl * np.cos(knee_angle))
    
    # 計算外展關節角度
    abduction_angle = - np.arctan2(y, z) + np.arctan2(h, L)

    return abduction_angle, hip_angle, knee_angle

def backward_kinematics_2d(x, z, hu, hl):
    """
    計算四足機器人腿部的關節角度（2D版本）。
    
    Parameters:
    x: 末端位置的x座標
    z: 末端位置的z座標
    hu: 大腿長度
    hl: 小腿長度
    
    Returns:
    hip_angle: 髖關節角度 (rad)
    knee_angle: 膝關節角度 (rad)
    """
    S = np.sqrt(np.square(x) + np.square(z))
    N = (np.square(S) - np.square(hl) - np.square(hu)) / (2 * hu * hl)
    N = np.clip(N, -1.0, 1.0)  # 確保 n 在 [-1, 1] 範圍
    knee_angle = -np.arccos(N)
    hip_angle = np.arctan2(x, -z) - np.arctan2(hl * np.sin(knee_angle), hu + hl * np.cos(knee_angle))

    return hip_angle, knee_angle


def forward_kinematics_2d(hip_angle, knee_angle, hu, hl):
    """
    計算四足機器人腿部末端位置（2D版本）。
    
    Parameters:
    hip_angle: 髖關節角度 (rad)
    knee_angle: 膝關節角度 (rad)
    hu: 大腿長度
    hl: 小腿長度
    
    Returns:
    x: 末端位置的x座標
    z: 末端位置的z座標
    """
    x = hu * np.sin(hip_angle) + hl * np.sin(hip_angle + knee_angle)
    z = -hu * np.cos(hip_angle) - hl * np.cos(hip_angle + knee_angle)
    return x, z