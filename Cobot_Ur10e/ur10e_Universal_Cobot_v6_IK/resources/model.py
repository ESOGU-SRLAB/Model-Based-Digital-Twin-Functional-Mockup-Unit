"""
UR10e Forward Kinematics FMU (Co‑Simulation)
===========================================
Pure kinematics – no dynamics, no states.

* **Inputs  (6)** : q1 … q6  (joint angles, **rad**)
* **Outputs (10)**: x, y, z (m)  – end‑effector position
                    roll, pitch, yaw (rad) – XYZ‑Euler
                    qx, qy, qz, qw         – quaternion
"""

import math
import numpy as np
from fmi2 import Fmi2FMU, Fmi2Status


# ─────────────────────────────────────────────────────────────
# Constants:
# ─────────────────────────────────────────────────────────────
# DH parameters (Craig's convention) for UR10e (mm, rad)
a1=math.pi/2 
a2=613.0 # mm
a3=572.0 # mm
d4=174.0 # mm
d5=120.0 # mm
LB, LTP = 181, 117  # mm (base offset, tool plate offset)

NUM_JOINTS = 6 # Number of joints

# Sabit dönüşümler – taban ofseti & tool‑plate + eksen swap
T_B0  = np.eye(4);  T_B0[3,4] = LB
T_6TP = np.eye(4);  T_6TP[3,4] = LTP
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# Utility functions:
# ─────────────────────────────────────────────────────────────
# Transformation matrix from ith frame to i-1th frame
def dh_transform_i_to_i_1(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ ct, -st,  0, a],
                     [ st*ca,  ct*ca, -sa, -d*sa],
                     [  st*sa,      ct*sa,     ca,   d*ca],
                     [  0,       0,      0,   1]])

# Transformation Matrix(Rotation Matrix) to RPY(ZYX) (detaylı incelenecek...)
def transformationMatrix_to_rpy(T):
    r11,r12,r13 = T[0][0], T[0][1], T[0][2]
    r21,r22,r23 = T[1][0], T[1][1], T[1][2]
    r31,r32,r33 = T[2][0], T[2][1], T[2][2]
    roll  = math.atan2(r21, r11)
    pitch = math.atan2(-r31, math.hypot(r32, r33))
    yaw   = math.atan2(r32, r33)
    return roll, pitch, yaw

# Transformation Matrix(Rotation Matrix) to Quaternion (detaylı incelenecek...)
def transformationMatrix_to_quaternion(T):
    m = T[:3,:3]; tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr+1.0)*2; qw = 0.25*s
        qx = (m[2,1]-m[1,2])/s; qy = (m[0,2]-m[2,0])/s; qz = (m[1,0]-m[0,1])/s
    else:
        i = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
        if i==0:
            s = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2
            qx = 0.25*s; qy=(m[0,1]+m[1,0])/s; qz=(m[0,2]+m[2,0])/s; qw=(m[2,1]-m[1,2])/s
        elif i==1:
            s = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2
            qx=(m[0,1]+m[1,0])/s; qy=0.25*s; qz=(m[1,2]+m[2,1])/s; qw=(m[0,2]-m[2,0])/s
        else:
            s = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2
            qx=(m[0,2]+m[2,0])/s; qy=(m[1,2]+m[2,1])/s; qz=0.25*s; qw=(m[1,0]-m[0,1])/s
    return qx, qy, qz, qw
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Model functions:
# ─────────────────────────────────────────────────────────────
# Forward Kinematics (FK) using Denavit-Hartenberg parameters (detaylı incelenecek...)
def forward_kinematics(q):
    """Return [x,y,z, roll,pitch,yaw, qx,qy,qz,qw] for joint vector q (rad)."""
    T = np.eye(4)
    for i,(a_prev,d_i,alpha_prev) in enumerate(DH_PARAMS):
        theta = q[i] + (math.pi/2 if i==1 else 0) - (math.pi/2 if i==3 else 0)
        T = T @ dh_transform(theta, d_i, a_prev, alpha_prev)
    T = T_B0 @ (T_fix @ T) @ T_6TP
    x,y,z               = T[0,3], T[1,3], T[2,3]
    roll,pitch,yaw      = matrix_to_rpy(T)
    qx,qy,qz,qw         = matrix_to_quaternion(T)
    return np.array([x,y,z, roll,pitch,yaw, qx,qy,qz,qw])
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 4) FMI Co‑Simulation class (stateless)
# Yukarıdaki yapıya göre güncellenecek.
# ─────────────────────────────────────────────────────────────
class Model(Fmi2FMU):
    """Stateless FK FMU for UR10e."""
    def __init__(self, reference_to_attr=None):
        super().__init__()
        self.reference_to_attr = reference_to_attr if reference_to_attr else {}
        self.q = np.zeros(NUM_JOINTS)
        self._update()
        default_map = {i: f"q{i+1}" for i in range(6)} | {
            6:"x",7:"y",8:"z", 9:"roll",10:"pitch",11:"yaw", 12:"qx",13:"qy",14:"qz",15:"qw"}
        self.reference_to_attr = {
            0: "q1",
            1: "q2",
            2: "q3",
            3: "q4",
            4: "q5",
            5: "q6",
            6: "x",
            7: "y",
            8: "z",
            9: "roll",
            10: "pitch",
            11: "yaw",
            12: "qx",
            13: "qy",
            14: "qz",
            15: "qw",
        }

    # pose recompute
    def _update(self):
        (self.x,self.y,self.z, self.roll,self.pitch,self.yaw, self.qx,self.qy,self.qz,self.qw) = forward_kinematics(self.q)
        

    def instantiate(self, instanceName, resourceLocation):
        return Fmi2Status.ok
    
    def setup_experiment(self, startTime, stopTime, tolerance):
        return Fmi2Status.ok
    
    def enter_initialization_mode(self):
        return Fmi2Status.ok
    
    def exit_initialization_mode(self):
        return Fmi2Status.ok
    
    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            attr = self.get_variable_name(ref)
            setattr(self, attr, val)
        return Fmi2Status.ok
    
    def get_real(self, refs):
        outvals = []
        for ref in refs:
            attr = self.get_variable_name(ref)
            outvals.append(getattr(self, attr))
        return outvals, Fmi2Status.ok

    def do_step(self,*_): return Fmi2Status.ok
    def terminate(self):
        return Fmi2Status.ok
    def reset(self):
        self.q[:] = 0.0; self._update(); return Fmi2Status.ok

def create_fmu_instance():
    return Model()
