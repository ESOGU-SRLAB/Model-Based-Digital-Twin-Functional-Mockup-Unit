# model.py (Inverse Kinematics FMU versiyonu)
import math
import numpy as np
import pickle
from fmi2 import Fmi2FMU, Fmi2Status

# ─────────────────────────────────────────────────────────────
# 1) Constants:
# ─────────────────────────────────────────────────────────────
a2, a3 = 0.613, 0.572   # Link lengths (m)
d4, d5 = 0.174, 0.120   # Link offsets (m)
LB, LTP = 0.181, 0.117  # Base lift (TB0) ve tool-plate offset (T6_TP)

NUM_JOINTS = 6

# TB0: Base → Link0
TB0 = np.eye(4)
TB0[2, 3] = LB

# T6_TP: Link6 → ToolPlate
T6_TP = np.eye(4)
T6_TP[2, 3] = LTP


# ─────────────────────────────────────────────────────────────
# 2) Utility Functions:
# ─────────────────────────────────────────────────────────────
def rpy_to_matrix(roll, pitch, yaw):
    """
    XYZ-Euler (roll, pitch, yaw) → 3×3 rotasyon matrisi.
    """
    cr = math.cos(roll);  sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw);   sy = math.sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[ cy, -sy, 0],
                   [ sy,  cy, 0],
                   [  0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,   0,    0],
                   [0,  cr, -sr],
                   [0,  sr,  cr]])
    return Rz @ Ry @ Rx


def inverse_kinematics(x_t, y_t, z_t, roll_t, pitch_t, yaw_t):
    """
    UR10e Inverse Kinematics.
    Inputs:
        x_t, y_t, z_t         : target end-effector position (m)
        roll_t, pitch_t, yaw_t: target end-effector XYZ-Euler angles (rad)
    Returns:
        q_solutions: (4×6) matrix, each row is a solution set [q1..q6]
                     (In this example, we will only return the first set.)
    """
    # (1) Build the 4x4 transformation matrix for the target:
    R_target = rpy_to_matrix(roll_t, pitch_t, yaw_t)
    T_B_TP = np.eye(4)
    T_B_TP[:3, :3] = R_target
    T_B_TP[0, 3] = x_t
    T_B_TP[1, 3] = y_t
    T_B_TP[2, 3] = z_t

    # (2) Compute {^0_6}T:
    #     TB0^-1 * T_B_TP * (T6_TP)^-1
    T0_6 = np.linalg.inv(TB0) @ T_B_TP @ np.linalg.inv(T6_TP)

    # T0_6 = [ [ r11 r12 r13 x6 ],
    #          [ r21 r22 r23 y6 ],
    #          [ r31 r32 r33 z6 ],
    #          [   0   0   0  1 ] ]

    r11, r12, r13, x6 = T0_6[0,0], T0_6[0,1], T0_6[0,2], T0_6[0,3]
    r21, r22, r23, y6 = T0_6[1,0], T0_6[1,1], T0_6[1,2], T0_6[1,3]
    r31, r32, r33, z6 = T0_6[2,0], T0_6[2,1], T0_6[2,2], T0_6[2,3]

    # (3) For θ1: E1 cosθ1 + F1 sinθ1 + G1 = 0
    E1 =  y6
    F1 = -x6
    G1 =  d4

    # t = tan(θ1/2) → quadratic: (G1 - E1) t^2 + (2 F1) t + (G1 + E1) = 0
    a_q1 = (G1 - E1)
    b_q1 = 2*F1
    c_q1 = (G1 + E1)
    disc1 = b_q1*b_q1 - 4*a_q1*c_q1
    if disc1 < 0:
        # Target is unreachable; No real solution.
        return None

    sqrt_disc1 = math.sqrt(disc1)
    t1a = (-b_q1 + sqrt_disc1) / (2*a_q1)
    t1b = (-b_q1 - sqrt_disc1) / (2*a_q1)
    θ1a = 2 * math.atan(t1a)
    θ1b = 2 * math.atan(t1b)

    solutions = []  # We will collect all (q1..q6) solution sets

    for θ1 in (θ1a, θ1b):
        c1 = math.cos(θ1);  s1 = math.sin(θ1)

        # (4) For θ6:  (r22 c1 - r12 s1) c6 + (r21 c1 - r11 s1) s6 = 0
        #  → tan θ6 = (r12 s1 - r22 c1) / (r21 c1 - r11 s1)
        num6 =  (r12*s1 - r22*c1)
        den6 =  (r21*c1 - r11*s1)
        θ6  = math.atan2(num6, den6)
        c6 = math.cos(θ6);   s6 = math.sin(θ6)

        # (5) For θ5: ratio of (2,1)/(2,2) terms:
        #  (r21 c1 - r11 s1) c6 - (r22 c1 - r12 s1) s6 = s5
        #  -r23 c1 + r13 s1 = c5
        s5 = (r21*c1 - r11*s1)*c6 - (r22*c1 - r12*s1)*s6
        c5 = -r23*c1 + r13*s1
        θ5 = math.atan2(s5, c5)

        # (6) Intermediate terms:
        #   A = s234 = (r31 c6 - r32 s6)/c5
        #   B = c234 = r32 c6 + r31 s6
        # Note: c5=cos(θ5); if c5≈0 there will be a division error. In practice,
        # this gives a small error; the target should be avoided or a c5 threshold should be checked.
        A = (r31*c6 - r32*s6) / c5
        B =  (r32*c6 + r31*s6)

        # (7) For θ2: E2 cosθ2 + F2 sinθ2 + G2 = 0
        #   a = - x6 c1 - y6 s1 - d5 * A
        #   b =  z6 - d5 * B
        a = -x6*c1 - y6*s1 - d5*A
        b =  z6 - d5*B
        E2 = -2 * a2 * b
        F2 = -2 * a2 * a
        G2 =  a2*a2 + a*a + b*b - a3*a3

        a_q2 = (G2 - E2)
        b_q2 = 2 * F2
        c_q2 = (G2 + E2)
        disc2 = b_q2*b_q2 - 4*a_q2*c_q2
        if disc2 < 0:
            continue  # no θ2 found for this θ1 → try next θ1

        sqrt_disc2 = math.sqrt(disc2)
        t2a = (-b_q2 + sqrt_disc2) / (2*a_q2)
        t2b = (-b_q2 - sqrt_disc2) / (2*a_q2)
        θ2a = 2 * math.atan(t2a)
        θ2b = 2 * math.atan(t2b)

        for θ2 in (θ2a, θ2b):
            c2 = math.cos(θ2);  s2 = math.sin(θ2)

            # (8) For θ3: ratio of two equations:
            #   a3 sin(θ2+θ3) = -a2 sin(θ2) + a
            #   a3 cos(θ2+θ3) = -a2 cos(θ2) + b
            num_23 = a - a2*s2
            den_23 = b - a2*c2
            θ23 = math.atan2(num_23, den_23)   # θ2+θ3
            θ3 = θ23 - θ2

            # (9) For θ4: atan2(A, B) - θ2 - θ3
            θ4 = math.atan2(A, B) - θ2 - θ3

            # Now we have a complete (θ1..θ6) set:
            solution = [θ1, θ2, θ3, θ4, θ5, θ6]
            solutions.append(solution)

    # There will be at most 4 sets in the “solutions” list.
    # For example, let's return only the “first” (elbow-up) set as output:
    if len(solutions) == 0:
        return None

    # Optionally, you can sort or classify the solutions here,
    # for example as "elbow up" / "elbow down" configurations.
    # In this example, we will only return solutions[0]:
    q_sol_1 = solutions[0]
    # FMU’daki set_output yapısına uygun 1×6 array dön:
    return np.array(q_sol_1)

class Model(Fmi2FMU):
    """
    FMI Co-Simulation model for UR10e Inverse Kinematics.
    Inputs  (valueRefs 0–5): x_t, y_t, z_t, roll_t, pitch_t, yaw_t
    Outputs (valueRefs 6–11): q1, q2, q3, q4, q5, q6
    """
    def __init__(self, reference_to_attr=None):
        super().__init__(reference_to_attr)
        # There are 12 variables in total: 6 inputs + 6 outputs
        # We will only keep the inputs and outputs:
        self.inputs = np.zeros(6)   # [x,y,z,roll,pitch,yaw]
        self.q_out  = np.zeros(NUM_JOINTS)  # [q1..q6]

        # valueReference → attribute name mapping:
        # 0–5 input, 6–11 output:
        self.reference_to_attr = {
            0: 'x_t',   1: 'y_t',   2: 'z_t',
            3: 'roll_t', 4: 'pitch_t', 5: 'yaw_t',
            6: 'q1', 7: 'q2', 8: 'q3',
            9: 'q4', 10: 'q5', 11: 'q6'
        }
        # Reset all attributes:
        for attr in self.reference_to_attr.values():
            setattr(self, attr, 0.0)

        # Update outputs at initialization:
        self._update_outputs()

    def _update_outputs(self):
        # Get from input attributes:
        x_t     = self.x_t
        y_t     = self.y_t
        z_t     = self.z_t
        roll_t  = self.roll_t
        pitch_t = self.pitch_t
        yaw_t   = self.yaw_t

        # Call Inverse Kinematics:
        q_sol = inverse_kinematics(x_t, y_t, z_t, roll_t, pitch_t, yaw_t)
        if q_sol is None:
            # For unreachable position, set all q's to NaN:
            self.q_out[:] = np.nan
        else:
            self.q_out[:] = q_sol

        # Write outputs to attributes:
        for i, name in enumerate(['q1','q2','q3','q4','q5','q6']):
            setattr(self, name, self.q_out[i])

    def serialize(self):
        # The model has no internal state (stateless), only Output/Inputs
        # can simply be returned as desired:
        data = pickle.dumps(self.q_out)
        return Fmi2Status.ok, data

    def deserialize(self, bytes_):
        self.q_out[:] = pickle.loads(bytes_)
        # (If desired, inputs can also be stored; simple example.)
        return Fmi2Status.ok

    def get_variable_name(self, vr):
        return self.reference_to_attr[vr]

    def set_real(self, refs, values):
        """
        'set' requests from outside the FMU are handled here.
        e.g.: refs = [0,3], values = [0.5, 1.2] etc.
        """
        for ref, val in zip(refs, values):
            attr = self.get_variable_name(ref)
            setattr(self, attr, val)
        return Fmi2Status.ok

    def get_real(self, refs):
        """
        'get' requests from outside the FMU are handled here:
          refs = [6,7,8,9,10,11] → [q1..q6] are requested
        """
        return [getattr(self, self.get_variable_name(r)) for r in refs], Fmi2Status.ok

    # All FMU lifecycle methods are the same as in the forward kinematics FMU:
    def instantiate(self, instanceName, resourceLocation): return Fmi2Status.ok
    def setup_experiment(self, startTime, stopTime, tolerance): return Fmi2Status.ok
    def enter_initialization_mode(self): return Fmi2Status.ok
    def exit_initialization_mode(self): return Fmi2Status.ok
    def do_step(self, current_time, step_size, no_prior):
        # _update_outputs will be called at each step:
        self._update_outputs()
        return Fmi2Status.ok
    def terminate(self): return Fmi2Status.ok
    def reset(self):
        # On reset, reset the inputs to zero:
        for i in range(6):
            setattr(self, self.reference_to_attr[i], 0.0)
        self._update_outputs()
        return Fmi2Status.ok

def create_fmu_instance():
    return Model()