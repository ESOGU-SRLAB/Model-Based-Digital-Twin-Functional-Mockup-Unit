"""
UR10e ‑ Forward Kinematics FMU (Co‑Simulation)
==============================================
* **Inputs  (6)** : q1 … q6  – joint angles in **radians**
* **Outputs (10)**: x, y, z (m) – end‑effector position  
                    roll, pitch, yaw (rad) – XYZ‑Euler  
                    qx, qy, qz, qw – quaternion (unit‑norm)

Model implements ONLY the forward‑kinematics chain published in
*Universal Robots e‑Series Analytical FPK* paper.  It includes:
  • Modified DH parameters for UR10e  
  • Built‑in frame fixes: Craig→UR DF axis swap, +LB base lift, +LTP tool‑plate offset.
No dynamics, control or internal states ⇒ single‑step algebraic FMU.
"""

import math
import numpy as np
from fmi2 import Fmi2FMU, Fmi2Status

# ──────────────────────────────────────────────────────────────
# 1.  Modified DH parameters (metres, radians) — Craig scheme
#     columns: (a_{i‑1}, d_i , alpha_{i‑1})
# ──────────────────────────────────────────────────────────────
DH_PARAMS = [
    (0.000,   0.000,  math.pi/2),   # i=1
    (0.000,   0.000,  math.pi/2),   # i=2  (θ₂ + 90° handled later)
    (0.613,   0.000,  0.0),         # i=3
    (0.572,   0.174,  0.0),         # i=4  (θ₄ − 90° handled later)
    (0.000,   0.120, -math.pi/2),   # i=5
    (0.000,   0.000,  math.pi/2)    # i=6
]
NUM_JOINTS = 6

# ──────────────────────────────────────────────────────────────
# 2.  Fixed transforms ({B}→{0} and {6}→{TP})
# ──────────────────────────────────────────────────────────────
LB   = 0.181   # base lift (m) – Table‑8
LTP  = 0.117   # tool‑plate length (m)

T_B0  = np.eye(4);  T_B0[2,3] = LB
T_6TP = np.eye(4);  T_6TP[2,3] = LTP
# Axis swap Craig {0}: X→Y, Y→‑X  (−π/2 around Z)
T_fix = np.array([[ 0, -1,  0, 0],
                  [ 1,  0,  0, 0],
                  [ 0,  0,  1, 0],
                  [ 0,  0,  0, 1]])

# ──────────────────────────────────────────────────────────────
# 3.  Kinematic helpers
# ──────────────────────────────────────────────────────────────

def dh_transform(theta: float, d: float, a: float, alpha: float):
    """4×4 DH homogeneous matrix."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ ct, -st*ca,  st*sa, a*ct],
                     [ st,  ct*ca, -ct*sa, a*st],
                     [  0,      sa,     ca,    d],
                     [  0,       0,      0,    1]])

def matrix_to_rpy(T):
    """XYZ‑Euler from rotation matrix."""
    r11, r12, r13 = T[0, :3]
    r21, r22, r23 = T[1, :3]
    r31, r32, r33 = T[2, :3]
    pitch = math.atan2(-r31, math.hypot(r11, r21))
    roll  = math.atan2(r21, r11)
    yaw   = math.atan2(r32, r33)
    return roll, pitch, yaw

def matrix_to_quaternion(T):
    """Convert rotation matrix to xyzw quaternion."""
    m = T[:3, :3]; tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2,1] - m[1,2]) / s
        qy = (m[0,2] - m[2,0]) / s
        qz = (m[1,0] - m[0,1]) / s
    else:
        idx = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
        if idx == 0:
            s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            qx = 0.25 * s
            qy = (m[0,1] + m[1,0]) / s
            qz = (m[0,2] + m[2,0]) / s
            qw = (m[2,1] - m[1,2]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            qx = (m[0,1] + m[1,0]) / s
            qy = 0.25 * s
            qz = (m[1,2] + m[2,1]) / s
            qw = (m[0,2] - m[2,0]) / s
        else:
            s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            qx = (m[0,2] + m[2,0]) / s
            qy = (m[1,2] + m[2,1]) / s
            qz = 0.25 * s
            qw = (m[1,0] - m[0,1]) / s
    return qx, qy, qz, qw


def forward_kinematics(q_rad):
    """Return pose vector [x,y,z, roll,pitch,yaw, qx,qy,qz,qw]."""
    T = np.eye(4)
    for i, (a_prev, d_i, alpha_prev) in enumerate(DH_PARAMS):
        theta = q_rad[i]
        if i == 1:   # θ₂ + 90°
            theta += math.pi/2
        elif i == 3: # θ₄ – 90°
            theta -= math.pi/2
        T = T @ dh_transform(theta, d_i, a_prev, alpha_prev)
    # sabit dönüştürmeler ve eksen düzeltmesi
    T_full = T_B0 @ (T_fix @ T) @ T_6TP
    x, y, z           = T_full[0,3], T_full[1,3], T_full[2,3]
    roll, pitch, yaw  = matrix_to_rpy(T_full)
    qx, qy, qz, qw    = matrix_to_quaternion(T_full)
    return np.array([x, y, z, roll, pitch, yaw, qx, qy, qz, qw])

# ──────────────────────────────────────────────────────────────
# 4.  FMI Co‑Simulation class
# ──────────────────────────────────────────────────────────────
class Model(Fmi2FMU):
    """Stateless FK FMU — UR10e"""
    def __init__(self):
        super().__init__()
        self.q = np.zeros(NUM_JOINTS)
        self._update_pose()
        # valueReference map
        self.ref = {i: f"q{i+1}" for i in range(6)}
        self.ref.update({
            6:"x",7:"y",8:"z",9:"roll",10:"pitch",11:"yaw",
            12:"qx",13:"qy",14:"qz",15:"qw"})

    # pose recompute
    def _update_pose(self):
        p = forward_kinematics(self.q)
        (self.x, self.y, self.z,
         self.roll, self.pitch, self.yaw,
         self.qx, self.qy, self.qz, self.qw) = p

    # FMI boilerplate (only essential)
    def instantiate(self, *a): return Fmi2Status.ok
    def setup_experiment(self, *a): return Fmi2Status.ok
    def enter_initialization_mode(self): return Fmi2Status.ok
    def exit_initialization_mode(self): self._update_pose(); return Fmi2Status.ok

    def set_real(self, refs, values):
        for r, v in zip(refs, values):
            if r < 6:
                self.q[r] = v
        self._update_pose()
        return Fmi2Status.ok

    def get_real(self, refs):
        return [getattr(self, self.ref[r]) for r in refs], Fmi2Status.ok

    def do_step(self, *a):
        return Fmi2Status.ok  # stateless

    def reset(self):
        self.q[:] = 0.0
        self._update_pose()
        return Fmi2Status.ok


def create_fmu_instance():
    return Model()
