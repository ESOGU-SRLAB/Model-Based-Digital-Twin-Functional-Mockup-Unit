import numpy as np
import matplotlib.pyplot as plt
import math
from fmi2 import Fmi2FMU, Fmi2Status

# ============================================================================
# Basic Constants and Limits
# ============================================================================
NUM_JOINTS = 6  # Number of joints in the UR10e robot
JOINT_VEL_LIMITS = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]      # Joint velocity limits (rad/s)
JOINT_TORQUE_LIMITS = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]  # Joint torque limits (Nm)

SAFE_Q_MIN = -math.pi
SAFE_Q_MAX = math.pi
SAFE_DQ_MAX = np.array(JOINT_VEL_LIMITS) * 0.9
SAFE_TAU_MAX = np.array(JOINT_TORQUE_LIMITS) * 0.8

def check_safety(q, dq, tau):
    """Check if joint positions, velocities, and torques are within safe limits."""
    safe = True
    for i in range(NUM_JOINTS):
        if not (SAFE_Q_MIN <= q[i] <= SAFE_Q_MAX):
            print(f"Safety violation: Joint {i+1} position {q[i]:.3f} out of range!")
            safe = False
        if abs(dq[i]) > SAFE_DQ_MAX[i]:
            print(f"Safety violation: Joint {i+1} velocity {dq[i]:.3f} exceeds safe limit!")
            safe = False
        if abs(tau[i]) > SAFE_TAU_MAX[i]:
            print(f"Safety violation: Joint {i+1} torque {tau[i]:.3f} exceeds safe limit!")
            safe = False
    return safe

# ============================================================================
# Performance Metrics Calculation Functions
# ============================================================================
def compute_rise_time(time, signal, target):
    lower_bound = 0.1 * target
    upper_bound = 0.9 * target
    try:
        t_low = time[np.where(signal >= lower_bound)[0][0]]
        t_high = time[np.where(signal >= upper_bound)[0][0]]
        return t_high - t_low
    except IndexError:
        return np.nan

def compute_overshoot(time, signal, target):
    max_val = np.max(signal)
    if max_val <= target:
        return 0.0
    return ((max_val - target) / target) * 100

def compute_settling_time(time, signal, target, tol=0.02):
    upper_bound = target * (1 + tol)
    lower_bound = target * (1 - tol)
    for idx in range(len(signal)):
        if np.all((signal[idx:] >= lower_bound) & (signal[idx:] <= upper_bound)):
            return time[idx]
    return np.nan

# -------------------------------
# Compute Desired Joint Angles from Inputs
# -------------------------------
def compute_desired_q(normal_inputs):
    """
    Computes the desired joint angles using the FMU's inverse kinematics.
    The desired end-effector pose is defined by the inputs.
    """
    des_pose = np.array([
        normal_inputs["x_des"],
        normal_inputs["y_des"],
        normal_inputs["z_des"],
        normal_inputs["roll_des"],
        normal_inputs["pitch_des"],
        normal_inputs["yaw_des"]
    ])
    # Initial guess is zeros (matching the FMU's initial state)
    q_des = inverse_kinematics_pose(des_pose, np.zeros(NUM_JOINTS))
    return q_des


# ============================================================================
# Kinematics Definitions
# ============================================================================
DH_PARAMS = [
    (0.1807,  0.0,     math.pi/2),
    (0.6127,  0.0,     0.0),
    (0.57155, 0.17415, 0.0),
    (0.11985, 0.0,     math.pi/2),
    (0.11655, 0.0,    -math.pi/2),
    (0.0,     0.0,     0.0)
]

LINK_MASSES = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])

def dh_transform(theta, d, a, alpha):
    """Create a 4x4 transformation matrix from DH parameters."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ ct,   -st*ca,  st*sa,  a*ct ],
        [ st,    ct*ca, -ct*sa,  a*st ],
        [  0,       sa,     ca,     d ],
        [  0,        0,      0,     1 ]
    ])

def forward_kinematics(q_rad):
    """Computes the end-effector pose [x, y, z, roll, pitch, yaw] from joint angles q_rad."""
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        theta_i = q_rad[i]
        a_i, d_i, alpha_i = DH_PARAMS[i]
        T_i = dh_transform(theta_i, d_i, a_i, alpha_i)
        T = T @ T_i
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    roll, pitch, yaw = matrix_to_rpy(T)
    return np.array([x, y, z, roll, pitch, yaw])

def matrix_to_rpy(T):
    """Convert a rotation matrix T to roll, pitch, yaw angles (X-Y-Z sequence)."""
    r11, r12, r13 = T[0,0], T[0,1], T[0,2]
    r21, r22, r23 = T[1,0], T[1,1], T[1,2]
    r31, r32, r33 = T[2,0], T[2,1], T[2,2]
    beta = math.atan2(-r31, math.sqrt(r11**2 + r21**2))
    alpha = math.atan2(r21, r11)
    gamma = math.atan2(r32, r33)
    return np.array([alpha, beta, gamma])

def compute_jacobian(q_rad, delta=1e-6):
    """Computes a numerical Jacobian of the end-effector pose with respect to joint angles."""
    base_pose = forward_kinematics(q_rad)
    J = np.zeros((6, NUM_JOINTS))
    for j in range(NUM_JOINTS):
        perturbed = q_rad.copy()
        perturbed[j] += delta
        pose_pert = forward_kinematics(perturbed)
        J[:, j] = (pose_pert - base_pose) / delta
    return J

def inverse_kinematics_pose(des_pose, init_q_rad, max_iter=100, alpha=0.01, tol=1e-4):
    """Iterative inverse kinematics to compute joint angles for a desired end-effector pose."""
    q = init_q_rad.copy()
    for _ in range(max_iter):
        current_pose = forward_kinematics(q)
        error = des_pose - current_pose
        if np.linalg.norm(error[:3]) < tol and np.linalg.norm(error[3:]) < tol:
            break
        J = compute_jacobian(q)
        J_inv = np.linalg.pinv(J, rcond=1e-3)
        q += alpha * (J_inv @ error)
    return q

# ============================================================================
# Dynamics Function (Newton-Euler Based)
# ============================================================================
def compute_dynamics(q, dq, ddq):
    """Computes joint torques using a simplified Newton-Euler approach."""
    tau = np.zeros(NUM_JOINTS)
    I_approx = np.array([0.05, 0.77, 0.31, 0.02, 0.02, 0.01])
    L_offset = 0.1
    FRICTION_COEFF = np.array([0.5, 0.5, 0.3, 0.2, 0.2, 0.1])
    for i in range(NUM_JOINTS):
        inertia_term = I_approx[i] * ddq[i]
        grav_term = LINK_MASSES[i] * 9.81 * L_offset * math.sin(q[i])
        friction_term = FRICTION_COEFF[i] * np.sign(dq[i]) if dq[i] != 0 else 0.0
        tau[i] = inertia_term + grav_term + friction_term
    return tau

# ============================================================================
# FMU Model Class Implementation
# ============================================================================
class Model(Fmi2FMU):
    """
    FMI Co-Simulation model for UR10e robot.
    Inputs (valueRefs 0-7):
      0: x_des, 1: y_des, 2: z_des,
      3: roll_des, 4: pitch_des, 5: yaw_des,
      6: kp, 7: kd
    Outputs (valueRefs):
      8-13: Joint angles q1...q6,
      14-19: Joint velocities dq1...dq6,
      20-25: End-effector pose [ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw],
      26-31: Joint torques tau1...tau6.
    """
    def __init__(self, reference_to_attr=None):
        super().__init__()
        self.reference_to_attr = reference_to_attr if reference_to_attr else {}
        self.q_rad = np.zeros(NUM_JOINTS)
        self.dq_rad = np.zeros(NUM_JOINTS)
        self.x_des = 0.5
        self.y_des = 0.0
        self.z_des = 0.5
        self.roll_des = 0.0
        self.pitch_des = 0.0
        self.yaw_des = 0.0
        self.kp = 10.0
        self.kd = 2.0
        self.q_out = np.zeros(NUM_JOINTS)
        self.dq_out = np.zeros(NUM_JOINTS)
        self.ee_pose = np.zeros(6)
        self.tau_out = np.zeros(NUM_JOINTS)
        self.reference_to_attr = {
            0: "x_des",
            1: "y_des",
            2: "z_des",
            3: "roll_des",
            4: "pitch_des",
            5: "yaw_des",
            6: "kp",
            7: "kd",
            8: "q1",
            9: "q2",
            10: "q3",
            11: "q4",
            12: "q5",
            13: "q6",
            14: "dq1",
            15: "dq2",
            16: "dq3",
            17: "dq4",
            18: "dq5",
            19: "dq6",
            20: "ee_x",
            21: "ee_y",
            22: "ee_z",
            23: "ee_roll",
            24: "ee_pitch",
            25: "ee_yaw",
            26: "tau1",
            27: "tau2",
            28: "tau3",
            29: "tau4",
            30: "tau5",
            31: "tau6"
        }
    
    def get_variable_name(self, ref):
        return self.reference_to_attr[ref]
    
    @property
    def q1(self): return self.q_out[0]
    @property
    def q2(self): return self.q_out[1]
    @property
    def q3(self): return self.q_out[2]
    @property
    def q4(self): return self.q_out[3]
    @property
    def q5(self): return self.q_out[4]
    @property
    def q6(self): return self.q_out[5]
    @property
    def dq1(self): return self.dq_out[0]
    @property
    def dq2(self): return self.dq_out[1]
    @property
    def dq3(self): return self.dq_out[2]
    @property
    def dq4(self): return self.dq_out[3]
    @property
    def dq5(self): return self.dq_out[4]
    @property
    def dq6(self): return self.dq_out[5]
    @property
    def ee_x(self): return self.ee_pose[0]
    @property
    def ee_y(self): return self.ee_pose[1]
    @property
    def ee_z(self): return self.ee_pose[2]
    @property
    def ee_roll(self): return self.ee_pose[3]
    @property
    def ee_pitch(self): return self.ee_pose[4]
    @property
    def ee_yaw(self): return self.ee_pose[5]
    @property
    def tau1(self): return self.tau_out[0]
    @property
    def tau2(self): return self.tau_out[1]
    @property
    def tau3(self): return self.tau_out[2]
    @property
    def tau4(self): return self.tau_out[3]
    @property
    def tau5(self): return self.tau_out[4]
    @property
    def tau6(self): return self.tau_out[5]
    
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
    
    def do_step(self, currentTime, stepSize, noSetFMUStatePriorToCurrentPoint):
        # 1) Inverse Kinematics
        des_pose = np.array([self.x_des, self.y_des, self.z_des,
                             self.roll_des, self.pitch_des, self.yaw_des])
        q_des = inverse_kinematics_pose(des_pose, self.q_rad)
        # 2) PD Control
        error = q_des - self.q_rad
        tau_cmd = self.kp * error - self.kd * self.dq_rad
        # 3) Compute Dynamics
        tau_dyn = compute_dynamics(self.q_rad, self.dq_rad, np.zeros(NUM_JOINTS))
        tau_eff = tau_cmd - tau_dyn
        # 4) Compute Joint Accelerations
        I_approx = np.array([0.05, 0.77, 0.31, 0.02, 0.02, 0.01])
        ddq = tau_eff / I_approx
        # 5) Euler Integration
        dq_new = self.dq_rad + ddq * stepSize
        q_new = self.q_rad + dq_new * stepSize
        for i in range(NUM_JOINTS):
            if abs(dq_new[i]) > JOINT_VEL_LIMITS[i]:
                dq_new[i] = np.sign(dq_new[i]) * JOINT_VEL_LIMITS[i]
            if abs(tau_cmd[i]) > JOINT_TORQUE_LIMITS[i]:
                tau_cmd[i] = np.sign(tau_cmd[i]) * JOINT_TORQUE_LIMITS[i]
        self.q_rad = q_new
        self.dq_rad = dq_new
        # 6) Update Outputs
        self.q_out = self.q_rad.copy()
        self.dq_out = self.dq_rad.copy()
        self.ee_pose = forward_kinematics(self.q_rad)
        self.tau_out = tau_cmd.copy()
        return Fmi2Status.ok
    
    def terminate(self):
        return Fmi2Status.ok
    
    def reset(self):
        self.q_rad = np.zeros(NUM_JOINTS)
        self.dq_rad = np.zeros(NUM_JOINTS)
        return Fmi2Status.ok

def create_fmu_instance():
    return Model()

# ============================================================================
# Simulation Code (Normal Conditions)
# ============================================================================
if __name__ == "__main__":
    # Simulation parameters
    T_total = 5.0   # Total simulation time (seconds)
    dt = 0.01       # Time step (seconds)
    num_steps = int(T_total / dt)

    # Set up FMU instance
    fmu = create_fmu_instance()
    fmu.instantiate("UR10e_FMU", "")
    fmu.setup_experiment(0, T_total, 1e-6)
    fmu.enter_initialization_mode()
    fmu.exit_initialization_mode()

    # Prepare arrays for logging data
    time_arr = np.linspace(0, T_total, num_steps)
    q_history = np.zeros((num_steps, NUM_JOINTS))
    dq_history = np.zeros((num_steps, NUM_JOINTS))
    tau_history = np.zeros((num_steps, NUM_JOINTS))
    safety_flag = np.ones(num_steps, dtype=bool)

    # Apply constant normal inputs and compute desired joint angles
    def apply_normal_inputs_wrapper(fmu):
        normal_inputs = {
            "x_des": 0.5,
            "y_des": 0.0,
            "z_des": 0.5,
            "roll_des": 0.0,
            "pitch_des": 0.0,
            "yaw_des": 0.0,
            "kp": 10.0,
            "kd": 2.0
        }
        input_refs = [0, 1, 2, 3, 4, 5, 6, 7]
        input_values = [
            normal_inputs["x_des"], normal_inputs["y_des"], normal_inputs["z_des"],
            normal_inputs["roll_des"], normal_inputs["pitch_des"], normal_inputs["yaw_des"],
            normal_inputs["kp"], normal_inputs["kd"]
        ]
        fmu.set_real(input_refs, input_values)
        return normal_inputs

    normal_inputs = apply_normal_inputs_wrapper(fmu)
    desired_q = compute_desired_q(normal_inputs)
    print("Computed Desired Joint Angles (Reference):", desired_q)

    # Simulation loop
    for i in range(num_steps):
        t = i * dt
        apply_normal_inputs_wrapper(fmu)
        fmu.do_step(t, dt, True)
        q_out, _ = fmu.get_real([8, 9, 10, 11, 12, 13])
        dq_out, _ = fmu.get_real([14, 15, 16, 17, 18, 19])
        tau_out, _ = fmu.get_real([26, 27, 28, 29, 30, 31])
        q_history[i, :] = q_out
        dq_history[i, :] = dq_out
        tau_history[i, :] = tau_out
        safety_flag[i] = check_safety(q_out, dq_out, tau_out)

    # Compute performance metrics
    metrics = {}
    for j in range(NUM_JOINTS):
        q_signal = q_history[:, j]
        target = desired_q[j]
        rise_time = compute_rise_time(time_arr, q_signal, target)
        overshoot = compute_overshoot(time_arr, q_signal, target)
        settling_time = compute_settling_time(time_arr, q_signal, target, tol=0.02)
        metrics[j+1] = {
            'Desired Target (rad)': target,
            'Rise Time (s)': rise_time,
            'Overshoot (%)': overshoot,
            'Settling Time (s)': settling_time
        }
    print("Performance Metrics under Normal Conditions:")
    for joint, vals in metrics.items():
        print(f"Joint {joint}: {vals}")

    # Plot step responses for each joint
    for j in range(NUM_JOINTS):
        plt.figure(figsize=(10, 6))
        plt.plot(time_arr, q_history[:, j], label=f'Joint {j+1} Angle')
        plt.axhline(desired_q[j], color='green', linestyle='--',
                    label=f'Desired Target ({desired_q[j]:.3f} rad)')
        lower_bound = 0.1 * desired_q[j]
        upper_bound = 0.9 * desired_q[j]
        try:
            t_low = time_arr[np.where(q_history[:, j] >= lower_bound)[0][0]]
            t_high = time_arr[np.where(q_history[:, j] >= upper_bound)[0][0]]
            plt.axvline(t_low, color='purple', linestyle=':', label='10% Threshold')
            plt.axvline(t_high, color='purple', linestyle='-.', label='90% Threshold')
            plt.text(t_low, lower_bound, f" t={t_low:.3f}s", color='purple')
            plt.text(t_high, upper_bound, f" t={t_high:.3f}s", color='purple')
        except IndexError:
            pass
        max_val = np.max(q_history[:, j])
        if max_val > desired_q[j]:
            t_max = time_arr[np.argmax(q_history[:, j])]
            plt.plot(t_max, max_val, 'ro', label='Max Overshoot')
            plt.text(t_max, max_val, f" {max_val:.3f} rad", color='red')
        st = metrics[j+1]['Settling Time (s)']
        if not np.isnan(st):
            plt.axvline(st, color='orange', linestyle='--', label='Settling Time')
            plt.text(st, desired_q[j], f" t={st:.3f}s", color='orange')
        plt.title(f'Joint {j+1} Step Response (Normal Conditions)')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [rad]')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    plt.figure(figsize=(12, 8))
    for j in range(NUM_JOINTS):
        plt.plot(time_arr, q_history[:, j], label=f'Joint {j+1}')
    plt.title('Joint Angles Over Time (Normal Conditions)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(time_arr, safety_flag, 'g-', label='Safety Status (1=Safe, 0=Violation)')
    plt.title('Safety Monitoring Over Time (Normal Conditions)')
    plt.xlabel('Time [s]')
    plt.ylabel('Safety')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()
