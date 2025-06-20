import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === 1. LOAD DATA ===
df1 = pd.read_excel(r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_joint_data_processed.xlsx")
df2 = pd.read_excel(r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\ur10e_outs.xlsx")
time = df1["time"]

# === 2. ANGULAR DIFFERENCE FUNCTION FOR EULER ANGLES ===
def angular_difference(a1, a2):
    diff = a1 - a2
    return (diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

# === 3. ERROR CALCULATION FUNCTION ===
def calculate_error_metrics(val1, val2, is_angle=False):
    if is_angle:
        diff = angular_difference(val1, val2)
        error = np.abs(diff)
    else:
        error = np.abs(val1 - val2)
    mae = np.mean(error)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean((val1 - val2) ** 2)) if not is_angle else np.sqrt(np.mean(diff ** 2))
    return error, mae, max_error, rmse

# === 4. DEFINE CATEGORIES ===
categories = {
    "Position (x, y, z)": ["x", "y", "z"],
    "Euler Angles (roll, pitch, yaw)": ["roll", "pitch", "yaw"],
    "Quaternions (qx, qy, qz, qw)": ["qx", "qy", "qz", "qw"]
}

# === 5. PDF EXPORT ===
with PdfPages(r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\RealUR10e-FMU_output_analysis.pdf") as pdf:
    plt.style.use("ggplot")
    for category_title, axes in categories.items():
        for axis in axes:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            val1 = df1[axis].values
            val2 = df2[axis].values

            is_angle = axis in ["roll", "pitch", "yaw"]
            error, mae, max_err, rmse = calculate_error_metrics(val1, val2, is_angle)

            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(time, val1, label=f'{axis} (Processed)', linewidth=2)
            ax1.plot(time, val2, label=f'{axis} (ur10e_out)', linestyle='--', linewidth=2)
            ax1.set_title(f"{category_title} - {axis} Comparison", fontsize=14)
            ax1.set_xlabel("Time (seconds)")
            if axis in ["x", "y", "z"]:
                ax1.set_ylabel(f"{axis} Value (meters)")
            else:
                ax1.set_ylabel(f"{axis} Value (radians)")     
            ax1.legend(loc="upper left")
            ax1.grid(True)
            plt.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # Plot 2: Error curve
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(time, error, label='Error (Absolute)', color='red', linewidth=1.5)
            ax2.set_title(f"{category_title} - {axis} Error\n"
                          f"MAE: {mae:.4f} | Max Error: {max_err:.4f} | RMSE: {rmse:.4f}", fontsize=14)
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Error (Absolute)")
            ax2.legend(loc="upper right")
            ax2.grid(True)

            # Mark the maximum error
            if max_err > 0.01:
                max_err_idx = np.argmax(error)
                ax2.axvline(time[max_err_idx], color='black', linestyle=':', alpha=0.7)
                ax2.text(time[max_err_idx], error[max_err_idx],
                         f'Max Error @ {time[max_err_idx]:.2f}s',
                         rotation=90, verticalalignment='bottom', fontsize=10)

            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

print("PDF successfully created with angle normalization: robot_comparison_analysis.pdf")
