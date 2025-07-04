# model.py

"""
UR10e Inverse Kinematics FMU (Co-Simulation)
============================================
This FMU implements the inverse kinematics for the UR10e robot arm.

* **Inputs  (6)** : 
                  x_t, y_t, z_t (m)  – end-effector position
                  roll_t, pitch_t, yaw_t (rad) – XYZ-Euler angles

* **Outputs (6)**: q1, q2, q3, q4, q5, q6 (rad) – joint angles
"""

import math
import numpy as np
import pickle
from fmi2 import Fmi2FMU, Fmi2Status

# ──────────────────────────────────────────────────
# 1) Constants:
# ──────────────────────────────────────────────────

d1 = 0.181
a2 = 0.613
a3 = 0.572
d4 = 0.174
d5 = 0.120
d6 = 0.117

# ──────────────────────────────────────────────────
# 2) Model Functions:
# ──────────────────────────────────────────────────

def solve_theta1(x, y, d4):
    E = -y
    F = x
    G = -d4
    a = G - E
    b = 2 * F
    c = G + E
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return []
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    return [2 * math.atan(t1), 2 * math.atan(t2)]

def solve_theta6(R, theta1):
    r11, r12 = R[0, 0], R[0, 1]
    r21, r22 = R[1, 0], R[1, 1]
    num = r12 * math.sin(theta1) - r22 * math.cos(theta1)
    den = r11 * math.cos(theta1) - r21 * math.sin(theta1)
    return math.atan2(num, den)

def solve_theta5(R, theta1, theta6):
    r13 = R[0, 2]
    r23 = R[1, 2]
    r33 = R[2, 2]
    sin_5 = r13 * math.cos(theta1) + r23 * math.sin(theta1)
    return math.atan2(sin_5, r33)

def solve_theta2_theta3(x, y, z, theta1, theta5):
    A = z - d1 - d4 * math.sin(theta5)
    B = x * math.cos(theta1) + y * math.sin(theta1) - d4 * math.cos(theta5)
    E = a2**2 + a3**2
    F = -2 * a2 * a3
    G = B**2 + A**2 - E
    a = G - E
    b = 2 * F
    c = G + E
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return []
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    theta2_1 = 2 * math.atan(t1)
    theta2_2 = 2 * math.atan(t2)
    def compute_theta3(theta2):
        k1 = a2 + a3 * math.cos(theta2)
        k2 = a3 * math.sin(theta2)
        return math.atan2(A, B) - math.atan2(k2, k1)
    return [(theta2_1, compute_theta3(theta2_1)), (theta2_2, compute_theta3(theta2_2))]

def solve_theta4(R, theta2, theta3, theta6):
    r31 = R[2, 0]
    r32 = R[2, 1]
    A = r32 * math.cos(theta6) - r31 * math.sin(theta6)
    B = -r32 * math.sin(theta6) - r31 * math.cos(theta6)
    theta234 = math.atan2(A, B)
    return theta234 - theta2 - theta3

def build_rotation_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def forward_kinematics(q):
    dh_params = [
        [math.pi/2, 0, d1],
        [0, -a2, 0],
        [0, -a3, 0],
        [math.pi/2, 0, d4],
        [-math.pi/2, 0, d5],
        [0, 0, d6]
    ]
    T = np.eye(4)
    for i in range(6):
        alpha, a, d = dh_params[i]
        theta = q[i]
        A = np.array([
            [math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
            [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
            [0, math.sin(alpha), math.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        T = T @ A
    return T[:3, 3], T[:3, :3]

def rotation_angle_error(R1, R2):
    R_diff = R1.T @ R2
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    return math.acos(trace)

def inverse_kinematics(x, y, z, roll, pitch, yaw):
    R_target = build_rotation_matrix(roll, pitch, yaw)
    solutions = []
    for theta1 in solve_theta1(x, y, d4):
        theta6 = solve_theta6(R_target, theta1)
        theta5 = solve_theta5(R_target, theta1, theta6)
        for theta2, theta3 in solve_theta2_theta3(x, y, z, theta1, theta5):
            theta4 = solve_theta4(R_target, theta2, theta3, theta6)
            q = [theta1, theta2, theta3, theta4, theta5, theta6]
            pos_fk, R_fk = forward_kinematics(q)
            pos_error = np.linalg.norm(pos_fk - np.array([x, y, z]))
            angle_error = rotation_angle_error(R_fk, R_target)
            total_error = pos_error + angle_error
            print("→ Candidate q (deg):", np.degrees(q), "→ total_error:", total_error)
            solutions.append((q, total_error))
    if not solutions:
        return []
    best_q = min(solutions, key=lambda item: item[1])[0]
    return best_q

# ──────────────────────────────────────────────────

class Model(Fmi2FMU):
    def __init__(self, reference_to_attr=None):
        super().__init__(reference_to_attr)
        self.reference_to_attr = {
            0: 'x_t', 1: 'y_t', 2: 'z_t',
            3: 'roll_t', 4: 'pitch_t', 5: 'yaw_t',
            6: 'q1', 7: 'q2', 8: 'q3', 9: 'q4', 10: 'q5', 11: 'q6'
        }
        for attr in self.reference_to_attr.values():
            setattr(self, attr, 0.0)

    def get_variable_name(self, vr):
        return self.reference_to_attr[vr]

    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            setattr(self, self.get_variable_name(ref), val)
        return Fmi2Status.ok

    def get_real(self, refs):
        return [getattr(self, self.get_variable_name(ref)) for ref in refs], Fmi2Status.ok

    def do_step(self, current_time, step_size, no_prior):
        print(f"[doStep] Inputs: x={self.x_t}, y={self.y_t}, z={self.z_t}, roll={self.roll_t}, pitch={self.pitch_t}, yaw={self.yaw_t}")
        q = inverse_kinematics(self.x_t, self.y_t, self.z_t, self.roll_t, self.pitch_t, self.yaw_t)
        if q:
            self.q1, self.q2, self.q3, self.q4, self.q5, self.q6 = q
        else:
            self.q1 = self.q2 = self.q3 = self.q4 = self.q5 = self.q6 = float('nan')
        return Fmi2Status.ok

    def serialize(self):
        return Fmi2Status.ok, pickle.dumps([
            self.x_t, self.y_t, self.z_t,
            self.roll_t, self.pitch_t, self.yaw_t
        ])

    def deserialize(self, state):
        self.x_t, self.y_t, self.z_t, self.roll_t, self.pitch_t, self.yaw_t = pickle.loads(state)
        return Fmi2Status.ok

    def instantiate(self, instanceName, resourceLocation): return Fmi2Status.ok
    def setup_experiment(self, startTime, stopTime, tolerance): return Fmi2Status.ok
    def enter_initialization_mode(self): return Fmi2Status.ok
    def exit_initialization_mode(self): return Fmi2Status.ok
    def terminate(self): return Fmi2Status.ok
    def reset(self): return Fmi2Status.ok

def create_fmu_instance():
    return Model()
