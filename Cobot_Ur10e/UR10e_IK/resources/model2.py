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
    print(f"[solve_theta1] Inputs: x={x}, y={y}, d4={d4}")
    E = -y
    F = x
    G = -d4
    a = G - E
    b = 2 * F
    c = G + E
    discriminant = b**2 - 4 * a * c
    print(f"[solve_theta1] a={a}, b={b}, c={c}, discriminant={discriminant}")
    if discriminant < 0:
        print("[solve_theta1] No real solution: discriminant < 0")
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

def inverse_kinematics(x, y, z, roll, pitch, yaw):
    """
    Compute inverse kinematics solutions for UR10e.
    Returns up to 4 solutions [[q1, ..., q6], ...]
    """
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
    R = Rz @ Ry @ Rx

    solutions = []
    for theta1 in solve_theta1(x, y, d4):
        theta6 = solve_theta6(R, theta1)
        theta5 = solve_theta5(R, theta1, theta6)
        for theta2, theta3 in solve_theta2_theta3(x, y, z, theta1, theta5):
            theta4 = solve_theta4(R, theta2, theta3, theta6)
            solutions.append([theta1, theta2, theta3, theta4, theta5, theta6])
    return solutions

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

        self.inputs = ['x_t', 'y_t', 'z_t', 'roll_t', 'pitch_t', 'yaw_t']
        self.outputs = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        self.initial_unknowns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']

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
        solutions = inverse_kinematics(self.x_t, self.y_t, self.z_t, self.roll_t, self.pitch_t, self.yaw_t)
        print(f"[doStep] Found {len(solutions)} IK solutions")

        if solutions:
            solution = solutions[0]
            self.q1, self.q2, self.q3, self.q4, self.q5, self.q6 = solution
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