import numpy as np
import matplotlib.pyplot as plt
import math
from model import create_fmu_instance, NUM_JOINTS, JOINT_VEL_LIMITS, JOINT_TORQUE_LIMITS

# -------------------------------
# External Safety Limits (for monitoring)
# -------------------------------
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

# -------------------------------
# Performance Metrics Calculation Functions
# -------------------------------
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
# Input Application Function (Non-Normal Conditions)
# -------------------------------
def apply_non_normal_inputs(fmu):
    """
    Applies the constant non-normal condition inputs to the FMU.
    Inputs (valueRefs 0-7):
      0: x_des, 1: y_des, 2: z_des,
      3: roll_des, 4: pitch_des, 5: yaw_des,
      6: kp, 7: kd
    """
    non_normal_inputs = {
        "x_des": 1.5,   # extreme x position
        "y_des": 0.0,
        "z_des": 1.5,
        "roll_des": 0.0,
        "pitch_des": 0.0,
        "yaw_des": 0.0,
        "kp": 20.0,     # higher gains may cause overshoot/oscillations
        "kd": 1.0       # lower damping increases instability risk
    }
    input_refs = [0, 1, 2, 3, 4, 5, 6, 7]
    input_values = [
        non_normal_inputs["x_des"], non_normal_inputs["y_des"], non_normal_inputs["z_des"],
        non_normal_inputs["roll_des"], non_normal_inputs["pitch_des"], non_normal_inputs["yaw_des"],
        non_normal_inputs["kp"], non_normal_inputs["kd"]
    ]
    fmu.set_real(input_refs, input_values)
    return non_normal_inputs  # Return dictionary for later use

# -------------------------------
# Simulation Setup
# -------------------------------
T_total = 5.0  # seconds
dt = 0.01      # time step [s]
num_steps = int(T_total / dt)

# Create FMU instance (using the unchanged model)
fmu = create_fmu_instance()
fmu.instantiate("UR10e_FMU", "")
fmu.setup_experiment(0, T_total, 1e-6)
fmu.enter_initialization_mode()
fmu.exit_initialization_mode()

# Prepare logging arrays
time_arr = np.linspace(0, T_total, num_steps)
q_history = np.zeros((num_steps, NUM_JOINTS))
dq_history = np.zeros((num_steps, NUM_JOINTS))
tau_history = np.zeros((num_steps, NUM_JOINTS))
safety_flag = np.ones(num_steps, dtype=bool)

# -------------------------------
# Simulation Loop: Non-Normal Conditions Only
# -------------------------------
for i in range(num_steps):
    t = i * dt
    # Apply non-normal constant inputs
    non_normal_inputs = apply_non_normal_inputs(fmu)
    
    # Perform simulation step
    fmu.do_step(t, dt, True)
    
    # Retrieve outputs
    q_out, _ = fmu.get_real([8, 9, 10, 11, 12, 13])
    dq_out, _ = fmu.get_real([14, 15, 16, 17, 18, 19])
    tau_out, _ = fmu.get_real([26, 27, 28, 29, 30, 31])
    
    q_history[i, :] = q_out
    dq_history[i, :] = dq_out
    tau_history[i, :] = tau_out
    
    # Safety check
    safety_flag[i] = check_safety(q_out, dq_out, tau_out)

# -------------------------------
# Compute Performance Metrics (Non-Normal Conditions)
# -------------------------------
metrics = {}
target_angle = 1.5  # For non-normal inputs, the target is 1.5 rad for x_des (assumed for all joints)
for j in range(NUM_JOINTS):
    q_signal = q_history[:, j]
    rise_time = compute_rise_time(time_arr, q_signal, target_angle)
    overshoot = compute_overshoot(time_arr, q_signal, target_angle)
    settling_time = compute_settling_time(time_arr, q_signal, target_angle, tol=0.02)
    metrics[j+1] = {
        'Rise Time (s)': rise_time,
        'Overshoot (%)': overshoot,
        'Settling Time (s)': settling_time
    }
print("Performance Metrics under Non-Normal Conditions:")
for joint, vals in metrics.items():
    print(f"Joint {joint}: {vals}")

# -------------------------------
# Plot Step Response for Each Joint (Non-Normal Conditions)
# -------------------------------
for j in range(NUM_JOINTS):
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, q_history[:, j], label=f'Joint {j+1} Angle')
    plt.axhline(target_angle, color='green', linestyle='--', label=f'Target ({target_angle} rad)')
    
    # Annotate 10% and 90% thresholds
    lower_bound = 0.1 * target_angle
    upper_bound = 0.9 * target_angle
    try:
        t_low = time_arr[np.where(q_history[:, j] >= lower_bound)[0][0]]
        t_high = time_arr[np.where(q_history[:, j] >= upper_bound)[0][0]]
        plt.axvline(t_low, color='purple', linestyle=':', label='10% Threshold')
        plt.axvline(t_high, color='purple', linestyle='-.', label='90% Threshold')
        plt.text(t_low, lower_bound, f" t={t_low:.3f}s", color='purple')
        plt.text(t_high, upper_bound, f" t={t_high:.3f}s", color='purple')
    except IndexError:
        pass
    
    # Annotate overshoot if present
    max_val = np.max(q_history[:, j])
    if max_val > target_angle:
        t_max = time_arr[np.argmax(q_history[:, j])]
        plt.plot(t_max, max_val, 'ro', label='Max Overshoot')
        plt.text(t_max, max_val, f" {max_val:.3f} rad", color='red')
    
    # Annotate settling time
    st = metrics[j+1]['Settling Time (s)']
    if not np.isnan(st):
        plt.axvline(st, color='orange', linestyle='--', label='Settling Time')
        plt.text(st, target_angle, f" t={st:.3f}s", color='orange')
    
    plt.title(f'Joint {j+1} Step Response (Non-Normal Conditions)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# -------------------------------
# Plot Overall Simulation Results and Safety Status (Non-Normal Conditions)
# -------------------------------
plt.figure(figsize=(12, 8))
for j in range(NUM_JOINTS):
    plt.plot(time_arr, q_history[:, j], label=f'Joint {j+1}')
plt.title('Joint Angles Over Time (Non-Normal Conditions)')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time_arr, safety_flag, 'g-', label='Safety Status (1=Safe, 0=Violation)')
plt.title('Safety Monitoring Over Time (Non-Normal Conditions)')
plt.xlabel('Time [s]')
plt.ylabel('Safety')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.show()
