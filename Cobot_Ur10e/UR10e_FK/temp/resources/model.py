# model.py

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
import pickle
from fmi2 import Fmi2FMU, Fmi2Status

# ─────────────────────────────────────────────────────────────
# 1) Constants:
# ─────────────────────────────────────────────────────────────
# Constants for UR10e
LB = 0.181    # Base height
a2 = 0.613    # Link 2 length
a3 = 0.572    # Link 3 length  
d4 = 0.174    # Wrist 1 offset
d5 = 0.120    # Wrist 2 offset
LTP = 0.117   # Tool plate offset
NUM_JOINTS = 6

# Pre-compute frequently used transforms
T_B0 = np.eye(4)
T_B0[2, 3] = LB
print(f"[FK] T_B0: {T_B0}")

T_6TP = np.eye(4)
T_6TP[2, 3] = LTP
print(f"[FK] T_6TP: {T_6TP}")

# ─────────────────────────────────────────────────────────────
# 2) Model Functions:
# ─────────────────────────────────────────────────────────────
def forward_kinematics(q):
    """
    The closed-form expressions for r_ij and ₀x₆,₀y₆,₀z₆ from the paper → T₀₆
    Then, we add the fixed transformations T_B0 and T_6TP to obtain T_BTP.
    """
    print(f"[FK] Joint Angles (before offset): {q}")
    
    # Apply DH parameter offsets as specified in Table 7
    θ1 = q[0]
    θ2 = q[1] + math.pi/2  # Apply -90° offset
    θ3 = q[2]
    θ4 = q[3] + math.pi/2  # Apply -90° offset
    θ5 = q[4]
    θ6 = q[5]
    
    print(f"[FK] Joint Angles (after offset): [{θ1:.4f}, {θ2:.4f}, {θ3:.4f}, {θ4:.4f}, {θ5:.4f}, {θ6:.4f}]")

    # Trigonometric calculations
    c1, s1 = math.cos(θ1), math.sin(θ1)
    c2, s2 = math.cos(θ2), math.sin(θ2)
    c3, s3 = math.cos(θ3), math.sin(θ3)
    c4, s4 = math.cos(θ4), math.sin(θ4)
    c5, s5 = math.cos(θ5), math.sin(θ5)
    c6, s6 = math.cos(θ6), math.sin(θ6)
    
    c23  = math.cos(θ2 + θ3)
    s23  = math.sin(θ2 + θ3)
    c234 = math.cos(θ2 + θ3 + θ4)
    s234 = math.sin(θ2 + θ3 + θ4)

    # Rotation matrix elements (from paper, with corrected angles)
    r11 = c1*(c234*c5*c6 - s234*s6) - s1*s5*c6
    r21 = s1*(c234*c5*c6 - s234*s6) + c1*s5*c6
    r31 = s234*c5*c6 + c234*s6
    
    r12 = -c1*(c234*c5*s6 + s234*c6) + s1*s5*s6
    r22 = -s1*(c234*c5*s6 + s234*c6) - c1*s5*s6
    r32 = -s234*c5*s6 + c234*c6
    
    
    r13 = c1*c234*s5 + s1*c5
    r23 = s1*c234*s5 - c1*c5
    r33 = s234*s5

    # Position from paper (these are for frame 6 origin)
    x6 = d4*s1 - c1*(a2*s2 + a3*s23 + d5*s234)
    y6 = -d4*c1 - s1*(a2*s2 + a3*s23 + d5*s234)
    z6 = a2*c2 + a3*c23 + d5*c234

    # Build T06
    T06 = np.array([
        [r11, r12, r13, x6],
        [r21, r22, r23, y6],
        [r31, r32, r33, z6],
        [  0,   0,   0,  1]
    ])
    
    print(f"[FK] T06:\n{T06}")
    
    # Final transformation: Base to Tool Plate
    T_BTP = T_B0 @ T06 @ T_6TP
    
    print(f"[FK] T_BTP:\n{T_BTP}")
    
    # Extract position
    x = T_BTP[0, 3]
    y = T_BTP[1, 3]
    z = T_BTP[2, 3]
    
    print(f"[FK] End-Effector Position: x={x:.4f}, y={y:.4f}, z={z:.4f}")
    
    # Extract orientation as RPY
    roll, pitch, yaw = matrix_to_rpy(T_BTP)
    print(f"[FK] Euler Angles: roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
    
    # Extract orientation as quaternion
    qx, qy, qz, qw = matrix_to_quaternion(T_BTP)
    print(f"[FK] Quaternion: qx={qx:.4f}, qy={qy:.4f}, qz={qz:.4f}, qw={qw:.4f}")

    return np.array([x, y, z, roll, pitch, yaw, qx, qy, qz, qw])
# ─────────────────────────────────────────────────────────────
# 3) Utility Functions:
# ─────────────────────────────────────────────────────────────
def matrix_to_rpy(T):
    """
    Converts a 4x4 homogeneous transformation matrix to XYZ Euler angles (roll, pitch, yaw).
    """
    R = T[:3, :3]
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else: # Gimbal lock singularity
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0
        
    return roll, pitch, yaw

def matrix_to_quaternion(T):
    """
    Converts a 4x4 homogeneous transformation matrix to a quaternion (qx, qy, qz, qw).
    This implementation is robust against numerical errors.
    """
    R = T[:3, :3]
    tr = np.trace(R) # R[0,0] + R[1,1] + R[2,2]

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
    

# ─────────────────────────────────────────────────────────────
# 4) FMU Co-Simulation Class:
# ─────────────────────────────────────────────────────────────
class Model(Fmi2FMU):
    """
    FMI Co-Simulation model for UR10e robot.
    Inputs (valueRefs 0-5):
        q1, q2, q3, q4, q5, q6 (joint angles in radians)
    Outputs (valueRefs 6-15):
        x, y, z (end-effector position in meters)
        roll, pitch, yaw (end-effector orientation in radians)
        qx, qy, qz, qw (quaternion representation of orientation)
    """
    def __init__(self, reference_to_attr=None):
        super().__init__(reference_to_attr)
        self.q = np.zeros(NUM_JOINTS)
        self.reference_to_attr = {
            0: 'q1', 1: 'q2', 2: 'q3', 3: 'q4', 4: 'q5', 5: 'q6',
            6: 'x',  7: 'y',  8: 'z',
            9: 'roll',  10: 'pitch', 11: 'yaw',
            12: 'qx', 13: 'qy', 14: 'qz', 15: 'qw'
        }
        for attr in self.reference_to_attr.values():
            setattr(self, attr, 0.0)
        self._update_outputs()

    def _update_outputs(self):
        q_vals = [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6]
        self.q[:] = q_vals
        result = forward_kinematics(self.q)
        for i, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw", "qx", "qy", "qz", "qw"]):
            setattr(self, name, result[i])

    def serialize(self):
        return Fmi2Status.ok, pickle.dumps(self.q)

    def deserialize(self, bytes):
        self.q = pickle.loads(bytes)
        self._update_outputs()
        return Fmi2Status.ok

    def get_variable_name(self, vr):
        return self.reference_to_attr[vr]

    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            attr = self.get_variable_name(ref)
            setattr(self, attr, val)
        return Fmi2Status.ok

    def get_real(self, refs):
        return [getattr(self, self.get_variable_name(ref)) for ref in refs], Fmi2Status.ok

    def instantiate(self, instanceName, resourceLocation): return Fmi2Status.ok
    def setup_experiment(self, startTime, stopTime, tolerance): return Fmi2Status.ok
    def enter_initialization_mode(self): return Fmi2Status.ok
    def exit_initialization_mode(self): return Fmi2Status.ok
    def terminate(self): return Fmi2Status.ok

    def do_step(self, current_time, step_size, no_prior):
        print(f"[do_step] Time={current_time:.3f}, Step={step_size:.3f}")
        self._update_outputs()
        return Fmi2Status.ok

    def reset(self):
        self.q[:] = 0.0
        self._update_outputs()
        return Fmi2Status.ok

def create_fmu_instance():
    return Model()

if __name__ == "__main__":
    print("Testing UR10e Forward Kinematics with corrected DH offsets...")
    pass