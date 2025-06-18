# model.py

"""
UR10e Forward Kinematics FMU (Co-Simulation)
============================================
Pure kinematics – no dynamics, no states.
This version is implemented according to the specific kinematic analysis
documents provided by the user. It uses the robust Denavit-Hartenberg (DH)
parameter method to ensure accurate and reliable calculations.

* **Inputs  (6)** : q1 … q6  (joint angles, **rad**)
* **Outputs (10)**: x, y, z (m)  – end-effector position
                  roll, pitch, yaw (rad) – XYZ-Euler
                  qx, qy, qz, qw        – quaternion
"""

import math
import numpy as np
import pickle
# from fmi2 import Fmi2FMU, Fmi2Status

# ─────────────────────────────────────────────────────────────
# 1) Constants:
# ─────────────────────────────────────────────────────────────
# Constants for UR10e as defined in "Tablo 1" of the provided document.
d1 = 0.181
a2 = 0.613
a3 = 0.572
d4 = 0.174
d5 = 0.120
d6 = 0.117  # This represents the length to the robot flange (Frame 6).
LTP = 0.0   # Tool plate offset (m). Set to a non-zero value for a custom tool.
NUM_JOINTS = 6

# ─────────────────────────────────────────────────────────────
# 2) Model Functions:
# ─────────────────────────────────────────────────────────────

def create_dh_transform_matrix(alpha, a, d, theta):
    """Creates a homogeneous transformation matrix from standard DH parameters."""
    # This is the standard formula for a transformation matrix A_{i-1}^{i}
    return np.array([
        [math.cos(theta), -math.sin(theta) * math.cos(alpha),  math.sin(theta) * math.sin(alpha), a * math.cos(theta)],
        [math.sin(theta),  math.cos(theta) * math.cos(alpha), -math.cos(theta) * math.sin(alpha), a * math.sin(theta)],
        [0,                math.sin(alpha),                    math.cos(alpha),                   d],
        [0,                0,                                  0,                                 1]
    ])

def forward_kinematics(q):
    """
    Calculates the forward kinematics for a UR10e robot based on the DH parameters
    provided in the user's kinematic analysis document ("Tablo 2").
    """
    # DH Parameters from "Tablo 2": [alpha_{i-1}, a_{i-1}, d_i, theta_offset]
    # We use the constants defined above. Note the signs for a2 and a3.
    dh_params = np.array([
        [math.pi/2,   0,    d1,   0],
        [0,          -a2,   0,    0],
        [0,          -a3,   0,    0],
        [math.pi/2,   0,    d4,   0],
        [-math.pi/2,  0,    d5,   0],
        [0,           0,    d6,   0]
    ])

    # Base Frame is Frame 0 as per the document.
    # Tool Frame can be offset from Frame 6 (robot flange).
    T_6_Tool = np.eye(4)
    T_6_Tool[2, 3] = LTP # Apply custom tool offset if any

    # Sequentially multiply transformation matrices for each joint: T_0^6 = A_1 * A_2 * ... * A_6
    T_0_6 = np.eye(4)
    for i in range(NUM_JOINTS):
        alpha, a, d, theta_offset = dh_params[i]
        # Per the document, no theta offsets are needed.
        theta = q[i] + theta_offset
        A_i = create_dh_transform_matrix(alpha, a, d, theta)
        T_0_6 = T_0_6 @ A_i

    # Final transformation from Base to Tooltip
    T_final = T_0_6 @ T_6_Tool

    # Extract position and orientation from the final transformation matrix
    x = T_final[0, 3]
    y = T_final[1, 3]
    z = T_final[2, 3]
    roll, pitch, yaw = matrix_to_rpy(T_final)
    
    qx, qy, qz, qw = matrix_to_quaternion(T_final)
    # ★ CANONICALIZATION: Dataset “qw ≥ 0” rule
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    return np.array([x, y, z, roll, pitch, yaw, qx, qy, qz, qw])

# ─────────────────────────────────────────────────────────────
# 3) Utility Functions:
# ─────────────────────────────────────────────────────────────
def matrix_to_rpy(T):
    """Converts a 4x4 transformation matrix to XYZ Euler angles (roll, pitch, yaw)."""
    R = T[:3, :3]
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0
        
    return roll, pitch, yaw

def matrix_to_quaternion(T):
    """Converts a 4x4 transformation matrix to a quaternion (qx, qy, qz, qw)."""
    R = T[:3, :3]
    tr = np.trace(R)

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
        
    return qx, qy, qz, qw

def get_all_joint_positions(q):
    """
    Returns the (x, y, z) positions of each joint (including end-effector).
    Output shape: (7, 3) → 7 frames (base to tooltip), each with (x, y, z)
    """
    dh_params = np.array([
        [math.pi/2,   0,    d1,   0],
        [0,          -a2,   0,    0],
        [0,          -a3,   0,    0],
        [math.pi/2,   0,    d4,   0],
        [-math.pi/2,  0,    d5,   0],
        [0,           0,    d6,   0]
    ])

    positions = []
    T = np.eye(4)
    positions.append(T[:3, 3])  # Base frame origin

    for i in range(NUM_JOINTS):
        alpha, a, d, theta_offset = dh_params[i]
        theta = q[i] + theta_offset
        A_i = create_dh_transform_matrix(alpha, a, d, theta)
        T = T @ A_i
        positions.append(T[:3, 3])

    return np.array(positions)
