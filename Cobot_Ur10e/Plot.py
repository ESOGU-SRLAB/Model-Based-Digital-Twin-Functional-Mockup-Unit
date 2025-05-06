import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ============================================================================
# Basic Constants and Limits
# ============================================================================
NUM_JOINTS = 6  # Number of joints in the UR10e robot
JOINT_VEL_LIMITS = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]      # (Not used here, but defined for completeness)
JOINT_TORQUE_LIMITS = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]  # (Not used in metric computation)

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

# ============================================================================
# Kinematics and Inverse Kinematics Definitions
# ============================================================================
DH_PARAMS = [
    (0.1807,  0.0,     math.pi/2),
    (0.6127,  0.0,     0.0),
    (0.57155, 0.17415, 0.0),
    (0.11985, 0.0,     math.pi/2),
    (0.11655, 0.0,    -math.pi/2),
    (0.0,     0.0,     0.0)
]

# (For computing desired joint angles; LINK_MASSES not used here but defined for completeness)
LINK_MASSES = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])

def dh_transform(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ ct,   -st*ca,  st*sa,  a*ct ],
        [ st,    ct*ca, -ct*sa,  a*st ],
        [  0,       sa,     ca,     d ],
        [  0,        0,      0,     1 ]
    ])

def matrix_to_rpy(T):
    r11, r12, r13 = T[0,0], T[0,1], T[0,2]
    r21, r22, r23 = T[1,0], T[1,1], T[1,2]
    r31, r32, r33 = T[2,0], T[2,1], T[2,2]
    beta = math.atan2(-r31, math.sqrt(r11**2 + r21**2))
    alpha = math.atan2(r21, r11)
    gamma = math.atan2(r32, r33)
    return np.array([alpha, beta, gamma])

def forward_kinematics(q_rad):
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        theta_i = q_rad[i]
        a_i, d_i, alpha_i = DH_PARAMS[i]
        T_i = dh_transform(theta_i, d_i, a_i, alpha_i)
        T = T @ T_i
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    roll, pitch, yaw = matrix_to_rpy(T)
    return np.array([x, y, z, roll, pitch, yaw])

def compute_jacobian(q_rad, delta=1e-6):
    base_pose = forward_kinematics(q_rad)
    J = np.zeros((6, NUM_JOINTS))
    for j in range(NUM_JOINTS):
        perturbed = q_rad.copy()
        perturbed[j] += delta
        pose_pert = forward_kinematics(perturbed)
        J[:, j] = (pose_pert - base_pose) / delta
    return J

def inverse_kinematics_pose(des_pose, init_q_rad, max_iter=100, alpha=0.01, tol=1e-4):
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
# Compute Desired Joint Angles Based on Normal Inputs
# ============================================================================
def compute_desired_q(normal_inputs):
    """
    Computes desired joint angles using the FMU's inverse kinematics based on the desired end-effector pose.
    """
    des_pose = np.array([
        normal_inputs["x_des"],
        normal_inputs["y_des"],
        normal_inputs["z_des"],
        normal_inputs["roll_des"],
        normal_inputs["pitch_des"],
        normal_inputs["yaw_des"]
    ])
    # Initial guess of zeros
    q_des = inverse_kinematics_pose(des_pose, np.zeros(NUM_JOINTS))
    return q_des

# ============================================================================
# Load Simulation Output Data from CSV
# ============================================================================
# Assume the CSV file has columns: "time", "q1", "q2", ..., "q6"
data = pd.read_csv("ur10e_Universal_Cobot_v4_out.csv")
time_arr = data["time"].to_numpy()
# Create a (num_steps x NUM_JOINTS) array for joint angles:
q_history = np.column_stack([data[f"q{i}"].to_numpy() for i in range(1, NUM_JOINTS+1)])

# ============================================================================
# Compute Desired Joint Angles (Reference)
# ============================================================================
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
desired_q = compute_desired_q(normal_inputs)
print("Computed Desired Joint Angles (Reference):", desired_q)

# ============================================================================
# Compute Performance Metrics for Each Joint
# ============================================================================
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

# ============================================================================
# Plot Step Response for Each Joint with Annotations
# ============================================================================
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
