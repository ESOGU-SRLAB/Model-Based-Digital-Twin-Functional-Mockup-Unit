import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 1) Read CSVs
# ==========================
real_path = r"/home/matisse/Documents/Targets_Torque.csv"              # Real robot
fmu_path  = r"/home/matisse/Documents/UR10e_InverseDynamics_out.csv"   # FMU output

real = pd.read_csv(real_path)
fmu  = pd.read_csv(fmu_path)

# ==========================
# 2) Map relevant columns
#    Targets_Torque: q1_tau_UR10 ... q6_tau_UR10  (real)
#    UR10e_InverseDynamics_out: tau1 ... tau6     (FMU)
# ==========================
real_cols = [f"q{i}_tau_UR10" for i in range(1, 7)]   # real robot torques
fmu_cols  = [f"tau{i}"        for i in range(1, 7)]   # FMU torques

# ==========================
# 3) Time / length alignment
#    In this dataset the FMU file has 1 extra sample; dropping the first row
#    gives a good alignment. Adjust this logic if other recordings differ.
# ==========================
if len(fmu) == len(real) + 1:
    fmu = fmu.iloc[1:].reset_index(drop=True)
elif len(real) == len(fmu) + 1:
    real = real.iloc[1:].reset_index(drop=True)
else:
    min_len = min(len(real), len(fmu))
    real = real.iloc[:min_len].reset_index(drop=True)
    fmu  = fmu.iloc[:min_len].reset_index(drop=True)

# Time axis (FMU file contains 'time' column)
if "time" in fmu.columns:
    t = fmu["time"].values
else:
    t = np.arange(len(real))

# ==========================
# 4) Error metrics and plots
# ==========================
results = []

for j in range(6):
    joint_id = j + 1

    real_joint = real[real_cols[j]].values   # real torque
    fmu_joint  = fmu[fmu_cols[j]].values     # FMU torque

    # Gap / error signal (FMU - real)
    err = fmu_joint - real_joint

    # Metrics
    rmse = np.sqrt(np.mean(err ** 2))
    mae  = np.mean(np.abs(err))
    bias = np.mean(err)                     # mean bias
    real_mean = np.mean(real_joint)
    real_std  = np.std(real_joint)
    nrmse_std = rmse / (real_std + 1e-9)    # normalized RMSE (by std)

    results.append({
        "joint": joint_id,
        "RMSE": rmse,
        "MAE": mae,
        "Bias(FMU-Real)": bias,
        "Real_Mean": real_mean,
        "Real_Std": real_std,
        "NRMSE_over_std": nrmse_std,
        "Min_Error": np.min(err),
        "Max_Error": np.max(err),
    })

    # --------------------------
    # Plot: separate figure per joint
    # --------------------------
    plt.figure(figsize=(10, 6))

    # Top: torques
    plt.subplot(2, 1, 1)
    plt.plot(t, real_joint, label="Real (Targets_Torque)")
    plt.plot(t, fmu_joint,  linestyle="--", label="FMU (UR10e_InverseDynamics_out)")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Joint {joint_id} Torque Comparison")
    plt.grid(True)
    plt.legend()

    # Bottom: error (gap)
    plt.subplot(2, 1, 2)
    plt.plot(t, err, label="Error (FMU - Real)")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Time [index / s]")
    plt.ylabel("Torque Error [Nm]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==========================
# 5) Results table
# ==========================
metrics_df = pd.DataFrame(results)
print("\n=== Joint-wise Error Metrics ===")
print(metrics_df.round(4))

# Optionally save as CSV
metrics_df.to_csv("torque_comparison_metrics.csv", index=False)
print("\nMetrics written to 'torque_comparison_metrics.csv'.")

