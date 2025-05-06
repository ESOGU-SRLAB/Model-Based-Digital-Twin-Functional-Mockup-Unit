# ur10e_model_v1.2.py

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
# 1) Constants:
# ─────────────────────────────────────────────────────────────
a2, a3 = 0.613, 0.572   # m
d4, d5 = 0.174, 0.120   # m
LB, LTP = 0.181, 0.117  # m (base lift, tool‐plate offset)
NUM_JOINTS = 6
# ─────────────────────────────────────────────────────────────
# 2) Analytical Forward Kinematics:
# ─────────────────────────────────────────────────────────────
def forward_kinematics(q):
    """
    Makaledeki kapalı form r_ij ve ₀x₆,₀y₆,₀z₆ ifadeleri → T₀₆
    Ardından TB0 ve T6TP sabit dönüşümleri ekleyerek TB_TP elde ediyoruz.
    """
    θ1, θ2, θ3, θ4, θ5, θ6 = q

    # trig ön‐bileşenler
    c1, s1 = math.cos(θ1), math.sin(θ1)
    c2, s2 = math.cos(θ2), math.sin(θ2)
    # c3,s3 aslında T₀₆’da kullanılmıyor doğrudan
    # c4,s4 de
    c5, s5 = math.cos(θ5), math.sin(θ5)
    c6, s6 = math.cos(θ6), math.sin(θ6)

    # birleşik açılar
    c23  = math.cos(θ2 + θ3)
    s23  = math.sin(θ2 + θ3)
    c234 = math.cos(θ2 + θ3 + θ4)
    s234 = math.sin(θ2 + θ3 + θ4)

    # rotasyon matrisinin analitik elemanları
    r11 = -s1*s5*s6 + c1*(-s234*s6 + c234*c5*c6)
    r21 =  c1*s5*s6 + s1*(-s234*s6 + c234*c5*c6)
    r31 =       c234*s6 + s234*c5*c6

    r12 =  s1*s5*c6 - c1*( s234*c6 + c234*c5*s6)
    r22 = -c1*s5*c6 - s1*( s234*c6 + c234*c5*s6)
    r32 =       c234*c6 - s234*c5*s6

    r13 =  s1*c5 + c1*c234*s5
    r23 = -c1*c5 + s1*c234*s5
    r33 =             s234*s5

    # T₀₆ öteleme bileşenleri
    x6 = d4*s1 - c1*(a2*s2 + a3*s23 + d5*s234)
    y6 = -d4*c1 - s1*(a2*s2 + a3*s23 + d5*s234)
    z6 =       a2*c2 + a3*c23 + d5*c234

    # önce TB0 (base lift), sonra T0₆, en son T6TP (tool offset)
    # homojen dönüşüm T_B_TP = TB0 @ T0_6 @ T6TP
    T06 = np.array([
        [r11, r12, r13, x6],
        [r21, r22, r23, y6],
        [r31, r32, r33, z6],
        [  0,   0,   0,  1]
    ])
    TB_TP = np.eye(4)
    TB_TP[2,3] = LB
    T6_TP = np.eye(4)
    T6_TP[2,3] = LTP

    T_final = TB_TP @ T06 @ T6_TP

    # pozisyon (metre)
    x, y, z = T_final[0,3], T_final[1,3], T_final[2,3]
    # oryantasyon
    roll, pitch, yaw = matrix_to_rpy(T_final)
    # quaternion
    qx, qy, qz, qw  = matrix_to_quaternion(T_final)

    return np.array([x, y, z, roll, pitch, yaw, qx, qy, qz, qw])

# ─────────────────────────────────────────────────────────────
# 3) RPY ve Quaternion çıkarımı:
# ─────────────────────────────────────────────────────────────
def matrix_to_rpy(T):
    r11, r12, r13 = T[0,:3]
    r21, r22, r23 = T[1,:3]
    r31, r32, r33 = T[2,:3]
    pitch = math.atan2(-r31, math.hypot(r11, r21))
    roll  = math.atan2(r21, r11)
    yaw   = math.atan2(r32, r33)
    return roll, pitch, yaw

def matrix_to_quaternion(T):
    m = T[:3,:3]; tr = np.trace(m)
    if tr > 0:
        s  = math.sqrt(tr+1.0)*2
        qw = 0.25*s
        qx = (m[2,1]-m[1,2]) / s
        qy = (m[0,2]-m[2,0]) / s
        qz = (m[1,0]-m[0,1]) / s
    else:
        i = int(np.argmax([m[0,0],m[1,1],m[2,2]]))
        if i==0:
            s  = math.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2
            qx = 0.25*s; qy=(m[0,1]+m[1,0])/s; qz=(m[0,2]+m[2,0])/s; qw=(m[2,1]-m[1,2])/s
        elif i==1:
            s  = math.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2
            qx=(m[0,1]+m[1,0])/s; qy=0.25*s; qz=(m[1,2]+m[2,1])/s; qw=(m[0,2]-m[2,0])/s
        else:
            s  = math.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2
            qx=(m[0,2]+m[2,0])/s; qy=(m[1,2]+m[2,1])/s; qz=0.25*s; qw=(m[1,0]-m[0,1])/s
    return qx, qy, qz, qw

# ─────────────────────────────────────────────────────────────
# 4) FMI Co-Simulation sınıfı:
# ─────────────────────────────────────────────────────────────
class Model(Fmi2FMU):
    """ur10e_model_v1.2: Stateless UR10e FK FMU with base & tool frames."""
    def __init__(self, reference_to_attr=None):
        super().__init__()
        self.q = np.zeros(6)
        self.reference_to_attr = reference_to_attr or {
            **{i: f"q{i+1}" for i in range(6)},
            **{6:"x",7:"y",8:"z",9:"roll",10:"pitch",11:"yaw",
               12:"qx",13:"qy",14:"qz",15:"qw"}
        }
        self._update()

    def _update(self):
        vals = forward_kinematics(self.q)
        (self.x,self.y,self.z,
         self.roll,self.pitch,self.yaw,
         self.qx,self.qy,self.qz,self.qw) = vals

    def instantiate(self,*a):              return Fmi2Status.ok
    def setup_experiment(self,*a):         return Fmi2Status.ok
    def enter_initialization_mode(self):   return Fmi2Status.ok
    def exit_initialization_mode(self):    self._update(); return Fmi2Status.ok

    def set_real(self, refs, vals):
        for r, v in zip(refs, vals):
            if r < 6: 
                self.q[r] = v  # Update q1 to q5
            elif r == 5:  # Update q6 if it's the 6th joint angle
                self.q[5] = v
        self._update()
        return Fmi2Status.ok


    def get_real(self, refs):
        out = []
        for r in refs:
            if r < NUM_JOINTS:
                out.append(self.q[r])
            else:
                out.append(getattr(self, self.reference_to_attr[r]))
        return out, Fmi2Status.ok


    def do_step(self,*a):                  return Fmi2Status.ok
    def terminate(self):                  return Fmi2Status.ok
    def reset(self):
        self.q[:] = 0
        self._update()
        return Fmi2Status.ok

def create_fmu_instance():
    return Model()
