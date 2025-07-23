""
""" 
UR10e Inverse Kinematics FMU (Co-Simulation)
=============================================
This FMU implements the inverse kinematics for the UR10e robot arm.

* **Inputs (6)** :
    x_t, y_t, z_t (m) – end-effector position
    roll_t, pitch_t, yaw_t (rad) – XYZ-Euler angles

* **Outputs (6)**: q1, q2, q3, q4, q5, q6 (rad) – joint angles

All 4 IK solution sets are calculated internally. Selection mechanism to be added later.
"""

import math
import numpy as np
import pickle
from fmi2 import Fmi2FMU, Fmi2Status

# ──────────────────────────────────────────────────────────
# 1) Constants:
# ──────────────────────────────────────────────────────────

# UR10e DH parameters
LB = 0.181
a2 = -0.613
a3 = - 0.572
d4 = 0.174
d5 = 0.120
LTP = 0.117

NUM_JOINTS = 6

# Transformation offsets for FK/IK consistency
T_0_B = np.eye(4)
T_TP_6 = np.eye(4)
T_TP_6[2, 3] = LTP  # Tooltip offset along Z6 axis

# ──────────────────────────────────────────────────────────
# 2) Model Functions:
# ──────────────────────────────────────────────────────────

def analytical_ik_solver(T_target):
    """
    Calculates all 4 inverse kinematics solution sets for UR10e robot.
    """
    solutions = []
    T_0_6 = T_0_B @ T_target @ np.linalg.inv(T_TP_6)

    px, py, pz = T_0_6[0, 3], T_0_6[1, 3], T_0_6[2, 3]
    r11, r12, r13 = T_0_6[0, 0], T_0_6[0, 1], T_0_6[0, 2]
    r21, r22, r23 = T_0_6[1, 0], T_0_6[1, 1], T_0_6[1, 2]
    r31, r32 = T_0_6[2, 0], T_0_6[2, 1]

    E1 = py
    F1 = -px
    G1 = d4
    D1 = E1**2 + F1**2 - G1**2
    if D1 < 0:
        return []

    t1_a = (-F1 + math.sqrt(D1)) / (G1 - E1) if abs(G1 - E1) > 1e-9 else float('inf')
    t1_b = (-F1 - math.sqrt(D1)) / (G1 - E1) if abs(G1 - E1) > 1e-9 else float('inf')
    theta1_sols = [2 * math.atan(t1_a), 2 * math.atan(t1_b)]
    theta1_sols = [t for t in theta1_sols if not math.isnan(t) and abs(t) <= math.pi]

    for t1 in theta1_sols:
        c1, s1 = math.cos(t1), math.sin(t1)

        num6 = -(r12 * s1 - r22 * c1)
        den6 = -(r21 * c1 - r11 * s1)
        t6 = math.atan2(num6, den6)
        c6, s6 = math.cos(t6), math.sin(t6)

        c5 = r13 * s1 - r23 * c1
        s5 = -((r21 * c1 - r11 * s1) * c6 - (r22 * c1 - r12 * s1) * s6)
        t5 = math.atan2(s5, c5)

        A = (r31 * c6 - r32 * s6) / math.cos(t5) if abs(math.cos(t5)) > 1e-6 else 0.0
        B = r32 * c6 + r31 * s6

        a = -px * c1 - py * s1 - d5 * A
        b = pz - d5 * B

        E2 = -2 * a2 * b
        F2 = -2 * a2 * a
        G2 = a2**2 + a**2 + b**2 - a3**2

        D2 = E2**2 + F2**2 - G2**2
        if D2 < 0:
            continue

        t2_1 = (-F2 + math.sqrt(D2)) / (G2 - E2) if abs(G2 - E2) > 1e-9 else float('inf')
        t2_2 = (-F2 - math.sqrt(D2)) / (G2 - E2) if abs(G2 - E2) > 1e-9 else float('inf')

        for t2 in [2 * math.atan(t2_1), 2 * math.atan(t2_2)]:
            if math.isnan(t2) or abs(t2) > math.pi:
                continue

            c2, s2 = math.cos(t2), math.sin(t2)
            s23 = (-a - a2 * s2) / a3
            c23 = (b - a2 * c2) / a3
            t3 = math.atan2(s23, c23) - t2
            t4 = math.atan2(A, B) - t2 - t3

            q = np.array([t1, t2, t3, t4, t5, t6])
            q = np.arctan2(np.sin(q), np.cos(q))
            solutions.append(q)

    return solutions

# ──────────────────────────────────────────────────────────
# 3) Utility Functions:
# ──────────────────────────────────────────────────────────

def rpy_to_matrix(roll, pitch, yaw, x, y, z):
    """Converts XYZ Euler angles and a position vector to a 4x4 transformation matrix."""
    c_r, s_r = math.cos(roll), math.sin(roll)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    R = np.array([
        [c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
        [-s_p,      c_p * s_r,                   c_p * c_r                  ]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# ──────────────────────────────────────────────────────────
# 4) FMU Co-Simulation Class:
# ──────────────────────────────────────────────────────────
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
        self.q_current = np.zeros(NUM_JOINTS)

    def _update_logic(self):
        T_target = rpy_to_matrix(self.roll_t, self.pitch_t, self.yaw_t,
                                 self.x_t, self.y_t, self.z_t)
        all_solutions = analytical_ik_solver(T_target)

        print("\n--- Computed IK Solutions (up to 4 sets) ---")
        for i, sol in enumerate(all_solutions):
            print(f"Solution {i+1}: {np.round(sol, 5)}")

        if all_solutions:
            selected_q = all_solutions[0]
            for i in range(NUM_JOINTS):
                setattr(self, f'q{i+1}', selected_q[i])
            self.q_current = selected_q

    def serialize(self):
        return Fmi2Status.ok, pickle.dumps(self.q_current)

    def deserialize(self, bytes_data):
        self.q_current = pickle.loads(bytes_data)
        return Fmi2Status.ok

    def get_variable_name(self, vr):
        return self.reference_to_attr[vr]

    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            setattr(self, self.get_variable_name(ref), val)
        return Fmi2Status.ok

    def get_real(self, refs):
        return [getattr(self, self.get_variable_name(ref)) for ref in refs], Fmi2Status.ok

    def enter_initialization_mode(self): self._update_logic(); return Fmi2Status.ok
    def exit_initialization_mode(self): self._update_logic(); return Fmi2Status.ok
    def do_step(self, current_time, step_size, no_set_fmu_state_prior_to_current_point): self._update_logic(); return Fmi2Status.ok
    def reset(self):
        for attr in self.reference_to_attr.values(): setattr(self, attr, 0.0)
        self.q_current = np.zeros(NUM_JOINTS)
        return Fmi2Status.ok
    def setup_experiment(self, start_time, stop_time=None, tolerance=None): return Fmi2Status.ok
    def terminate(self): return Fmi2Status.ok

# ──────────────────────────────────────────────────────────
# 5) Model Execution Block:
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing UR10e Inverse Kinematics model stub...")
""
