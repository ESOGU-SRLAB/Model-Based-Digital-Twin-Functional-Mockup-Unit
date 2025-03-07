import numpy as np
from fmi2 import Fmi2FMU, Fmi2Status
import math

###############################################################################
# RS005L Robot Data
###############################################################################

NUM_JOINTS = 6

# DH parameters: (theta_offset_deg, d, a, alpha_deg)
DH_PARAMS = [
    (0,    0.285, 0,   90),
    (0,    0.105, 0,  -90),
    (0,    0.380, 0,   90),
    (0,    0.080, 0,  -90),
    (0,    0.143, 0,   90),
    (0,    0.267, 0,  -90)
]

def dh_transform(theta_deg, d, a, alpha_deg):
    """Create a 4x4 DH transformation (angles in degrees)."""
    theta = math.radians(theta_deg)
    alpha = math.radians(alpha_deg)
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ ct,    -st*ca,  st*sa,  a*ct ],
        [ st,     ct*ca, -ct*sa,  a*st ],
        [  0,        sa,     ca,     d ],
        [  0,         0,      0,     1 ]
    ])

def forward_kinematics(joint_angles_deg):
    """Return end-effector [x,y,z]. joint_angles_deg is a 6-element array in degrees."""
    T = np.eye(4)
    for i, (theta0, d, a, alpha) in enumerate(DH_PARAMS):
        total_theta = theta0 + joint_angles_deg[i]
        T_i = dh_transform(total_theta, d, a, alpha)
        T = T @ T_i
    return T[:3, 3]

def compute_jacobian(joint_angles_deg, delta=1e-5):
    """
    Numerical Jacobian for the end-effector pos wrt each joint angle.
    3x6 matrix.
    """
    base_pos = forward_kinematics(joint_angles_deg)
    J = np.zeros((3, NUM_JOINTS))
    for j in range(NUM_JOINTS):
        perturbed = joint_angles_deg.copy()
        perturbed[j] += delta
        pos_pert = forward_kinematics(perturbed)
        J[:, j] = (pos_pert - base_pos) / delta
    return J

def inverse_kinematics_xyz(target_pos, init_q_deg, max_iter=100, alpha=0.01, tol=1e-4):
    """
    Very simple iterative IK to match [x,y,z]. Returns joint angles in deg.
    """
    q = init_q_deg.copy()
    for _ in range(max_iter):
        current_pos = forward_kinematics(q)
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            break
        J = compute_jacobian(q)
        J_inv = np.linalg.pinv(J, rcond=1e-3)
        q += alpha * (J_inv @ error)
    return q

# Basic dynamic placeholders
link_masses = np.array([10.395, 8.788, 7.575, 2.679, 1.028, 1.000])
link_lengths = [0.3, 0.3, 0.3, 0.2, 0.1, 0.1]
GRAVITY = 9.81

# approximate diagonal inertia
I_diag = [0.07, 0.05, 0.05, 0.02, 0.01, 0.01]  # or more advanced approach if you prefer

###############################################################################
# The FMU class
###############################################################################

class RobotFMU(Fmi2FMU):
    """
    Implements an FMI Co-Simulation model for the RS005L robot,
    combining a simple IK approach + dynamic step.

    The modelDescription.xml has assigned:
      - valueRef=0..4 => inputs: des_x, des_y, des_z, kp, kd
      - valueRef=5..10 => joint_angle_1..6
      - valueRef=11..13 => ee_x, ee_y, ee_z
    """

    def __init__(self):
        super().__init__()
        
        # INTERNAL STATES:
        # We'll keep q_deg, dq_deg as arrays for the ODE integration. (Degrees)
        self.q_deg = np.zeros(NUM_JOINTS)
        self.dq_deg = np.zeros(NUM_JOINTS)

        # Robot inputs (with some defaults)
        self.des_x = 0.3
        self.des_y = -0.2
        self.des_z = 0.5
        self.kp    = 10.0
        self.kd    = 2.0

        # Outputs: we'll store them as separate attributes, but they come from q_deg
        self.joint_angle_1 = 0.0
        self.joint_angle_2 = 0.0
        self.joint_angle_3 = 0.0
        self.joint_angle_4 = 0.0
        self.joint_angle_5 = 0.0
        self.joint_angle_6 = 0.0
        self.ee_x = 0.0
        self.ee_y = 0.0
        self.ee_z = 0.0

        # This dictionary maps the valueReferences from modelDescription.xml to attribute names in Python.
        # Must match the references you assigned in the XML.
        self.reference_to_attribute = {
            # inputs
            0: "des_x",
            1: "des_y",
            2: "des_z",
            3: "kp",
            4: "kd",
            # outputs
            5:  "joint_angle_1",
            6:  "joint_angle_2",
            7:  "joint_angle_3",
            8:  "joint_angle_4",
            9:  "joint_angle_5",
            10: "joint_angle_6",
            11: "ee_x",
            12: "ee_y",
            13: "ee_z"
        }
        
    ###########################################################################
    #  Overridden FMI2 lifecycle methods
    ###########################################################################
    def get_variable_name(self, ref):
        return self.reference_to_attribute[ref]

    def instantiate(self, instanceName, resourceLocation):
        return Fmi2Status.ok

    def setup_experiment(self, startTime, stopTime, tolerance):
        # Reset or initialize states if needed
        return Fmi2Status.ok
    
    def enter_initialization_mode(self):
        return Fmi2Status.ok

    def exit_initialization_mode(self):
        # Optionally do computations once all init inputs are known
        return Fmi2Status.ok
    
    def set_real(self, refs, values):
        """
        The master is telling us to set certain input variables (like des_x, kp, etc.).
        We'll map them to the correct Python attribute using reference_to_attribute.
        """
        for ref, val in zip(refs, values):
            attr_name = self.get_variable_name(ref)
            setattr(self, attr_name, val)
        return Fmi2Status.ok

    def get_real(self, refs):
        """
        The master is querying output variables (joint angles, end-effector, etc.).
        We'll read from the corresponding attributes using reference_to_attribute.
        """
        outvals = []
        for ref in refs:
            attr_name = self.get_variable_name(ref)
            val = getattr(self, attr_name)
            outvals.append(val)
        return outvals, Fmi2Status.ok

    def do_step(self, currentTime, stepSize, noSetFMUStatePriorToCurrentPoint):
        """
        This is called each co-simulation step. We combine:
          1) IK to get q_des from (des_x, des_y, des_z).
          2) PD torque in "degrees space".
          3) Gravity approximation => mg * sin(q_i).
          4) Forward-Euler integration.
          5) Update the output variables (joint_angle_1..6, ee_x..z).
        """

        # 1) IK
        target_pos = np.array([self.des_x, self.des_y, self.des_z])
        q_des_deg = inverse_kinematics_xyz(target_pos, self.q_deg)

        # 2) PD torque in deg-based
        # tau_i = kp * (q_des - q) - kd * dq
        error_deg = q_des_deg - self.q_deg
        tau_pd = self.kp * error_deg - self.kd * self.dq_deg

        # 3) Gravity torque (approx). We'll do mg*(L/2)*sin(q_i in rad)
        tau_g = np.zeros(NUM_JOINTS)
        for i in range(NUM_JOINTS):
            mg = link_masses[i] * GRAVITY * (link_lengths[i]*0.5)
            rad = math.radians(self.q_deg[i])
            tau_g[i] = mg * math.sin(rad)

        # net torque
        net_tau = tau_pd - tau_g

        # 4) ddq = net_tau / I_diag => integrate
        ddq_deg = net_tau / np.array(I_diag)  # still "degrees-based"
        self.q_deg  += self.dq_deg * stepSize
        self.dq_deg += ddq_deg * stepSize

        # 5) Now we reflect the internal q_deg array into the separate output attributes:
        self.joint_angle_1 = self.q_deg[0]
        self.joint_angle_2 = self.q_deg[1]
        self.joint_angle_3 = self.q_deg[2]
        self.joint_angle_4 = self.q_deg[3]
        self.joint_angle_5 = self.q_deg[4]
        self.joint_angle_6 = self.q_deg[5]

        # Also compute end-effector from new q_deg
        ee_pos = forward_kinematics(self.q_deg)
        self.ee_x = ee_pos[0]
        self.ee_y = ee_pos[1]
        self.ee_z = ee_pos[2]

        return Fmi2Status.ok

    def terminate(self):
        return Fmi2Status.ok

    def reset(self):
        """
        If the master calls reset, we can re-initialize everything to default.
        """
        self.q_deg = np.zeros(NUM_JOINTS)
        self.dq_deg = np.zeros(NUM_JOINTS)
        return Fmi2Status.ok

    ###########################################################################
    # The unifmu framework calls `get_variable_name(ref)`.
    # It uses self.reference_to_attribute if you haven't overridden that method.
    ###########################################################################


# The "entry point" for unifmu
def create_fmu_instance():
    return RobotFMU()
