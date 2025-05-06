"""
UR10e Forward Kinematics FMU (Co-Simulation)
============================================
Pure kinematics – no dynamics, no states.

* **Inputs  (6)** : q1 … q6  (joint angles, **rad**)
* **Outputs (10)**: x, y, z (m)  – end-effector position
                    roll, pitch, yaw (rad) – XYZ-Euler
                    qx, qy, qz, qw         – quaternion
"""

import math
import numpy as np
from fmi2 import Fmi2FMU, Fmi2Status

# ─────────────────────────────────────────────────────────────
# Constants:
# ─────────────────────────────────────────────────────────────
# DH parameters (Craig's convention) for UR10e (m, rad)
# (a_{i-1}, d_i, alpha_{i-1})
DH_PARAMS = [
    (0.000, 0.000,  math.pi/2),   # i=1
    (0.000, 0.000,  math.pi/2),   # i=2 (θ2 + 90°)
    (0.613, 0.000,  0.0),         # i=3 (a2)
    (0.572, 0.174,  0.0),         # i=4 (a3, d4; θ4 − 90°)
    (0.000, 0.120, -math.pi/2),   # i=5 (d5)
    (0.000, 0.000,  math.pi/2)    # i=6
]
NUM_JOINTS = 6

# Specific DH lengths (Table-8)
LB, LTP = 0.181, 0.117  # m

# Fixed transforms – base lift, tool-plate offset, axis swap (Craig→UR)
T_B0  = np.eye(4); T_B0[2,3]  = LB
T_6TP = np.eye(4); T_6TP[2,3] = LTP
T_fix = np.array([[ 0, -1, 0, 0],
                  [ 1,  0, 0, 0],
                  [ 0,  0, 1, 0],
                  [ 0,  0, 0, 1]])

# ─────────────────────────────────────────────────────────────
# Utility functions:
# ─────────────────────────────────────────────────────────────
def dh_transform(theta, d, a, alpha):
    """Craig-modified DH→4×4 homogeneous matrix."""
    """ In here, parameters are defined as follows:
    c_i = ct,
    s_i = st, 
    c_{\alpha_{i-1}} = ca,
    s_{\alpha_{i-1}} = sa,
    a_{i-1} = a, d_i = d
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ ct,  -st*ca,  st*sa,  a*ct],
        [ st,   ct*ca, -ct*sa,  a*st],
        [  0,      sa,     ca,     d],
        [  0,       0,      0,     1]
    ])

def matrix_to_rpy(T):
    """Rotation matrix→roll-pitch-yaw (XYZ)."""
    r11, r12, r13 = T[0,:3]
    r21, r22, r23 = T[1,:3]
    r31, r32, r33 = T[2,:3]
    pitch = math.atan2(-r31, math.hypot(r11, r21))
    roll  = math.atan2(r21, r11)
    yaw   = math.atan2(r32, r33)
    return roll, pitch, yaw

def matrix_to_quaternion(T):
    """Rotation matrix→(qx,qy,qz,qw)."""
    m = T[:3,:3]; tr = np.trace(m)
    if tr > 0:
        s  = math.sqrt(tr+1.0)*2
        qw = 0.25*s
        qx = (m[2,1]-m[1,2]) / s
        qy = (m[0,2]-m[2,0]) / s
        qz = (m[1,0]-m[0,1]) / s
    else:
        i = int(np.argmax([m[0,0],m[1,1],m[2,2]]))
        if i == 0:
            s  = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2]) * 2
            qx = 0.25*s
            qy = (m[0,1]+m[1,0]) / s
            qz = (m[0,2]+m[2,0]) / s
            qw = (m[2,1]-m[1,2]) / s
        elif i == 1:
            s  = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2]) * 2
            qx = (m[0,1]+m[1,0]) / s
            qy = 0.25*s
            qz = (m[1,2]+m[2,1]) / s
            qw = (m[0,2]-m[2,0]) / s
        else:
            s  = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1]) * 2
            qx = (m[0,2]+m[2,0]) / s
            qy = (m[1,2]+m[2,1]) / s
            qz = 0.25*s
            qw = (m[1,0]-m[0,1]) / s
    return qx, qy, qz, qw

# ─────────────────────────────────────────────────────────────
# Model functions:
# ─────────────────────────────────────────────────────────────
def forward_kinematics(q):
    """
    Return [x,y,z, roll,pitch,yaw, qx,qy,qz,qw] for q (rad).
    Implements R1…R8 steps: DH, chain, sin/cos sums, LB/LTP, axis-swap.
    """
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(DH_PARAMS):
        theta = q[i]
        if i == 1:
            theta += math.pi/2
        elif i == 3:
            theta -= math.pi/2
        T = T @ dh_transform(theta, d, a, alpha)
    # axis fix + base & tool offsets
    T = T_B0 @ (T_fix @ T) @ T_6TP

    # extract pose
    x, y, z      = T[0,3], T[1,3], T[2,3]
    roll,pitch,yaw = matrix_to_rpy(T)
    qx,qy,qz,qw  = matrix_to_quaternion(T)
    return np.array([x, y, z, roll, pitch, yaw, qx, qy, qz, qw])

# ─────────────────────────────────────────────────────────────
# 4) FMI Co-Simulation class (stateless)
# ─────────────────────────────────────────────────────────────
class Model(Fmi2FMU):
    """Stateless FK FMU for UR10e."""

    def __init__(self, reference_to_attr=None):
        super().__init__()
        # joint state
        self.q = np.zeros(NUM_JOINTS)
        # map VRef→attribute
        self.reference_to_attr = reference_to_attr or {
            **{i: f"q{i+1}" for i in range(6)},
            6:"x",7:"y",8:"z",
            9:"roll",10:"pitch",11:"yaw",
            12:"qx",13:"qy",14:"qz",15:"qw"
        }
        # initial pose
        self._update()

    def _update(self):
        (self.x,self.y,self.z,
         self.roll,self.pitch,self.yaw,
         self.qx,self.qy,self.qz,self.qw) = forward_kinematics(self.q)

    # FMI minimal interface
    def instantiate(self, *args):                 return Fmi2Status.ok
    def setup_experiment(self, *args):            return Fmi2Status.ok
    def enter_initialization_mode(self):          return Fmi2Status.ok
    def exit_initialization_mode(self):           self._update(); return Fmi2Status.ok

    def set_real(self, refs, values):
        for r,v in zip(refs, values):
            if r < NUM_JOINTS:
                self.q[r] = v
        self._update()
        return Fmi2Status.ok

    def get_real(self, refs):
        out = [getattr(self, self.reference_to_attr[r]) for r in refs]
        return out, Fmi2Status.ok

    def do_step(self, *args):                     return Fmi2Status.ok
    def reset(self):
        self.q[:] = 0.0
        self._update()
        return Fmi2Status.ok

def create_fmu_instance():
    return Model()
