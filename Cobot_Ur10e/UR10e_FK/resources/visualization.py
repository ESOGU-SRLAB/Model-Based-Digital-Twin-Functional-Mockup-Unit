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
with PdfPages(r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_comparison_analysis.pdf") as pdf:
    plt.style.use("ggplot")
    for category_title, axes in categories.items():
        for axis in axes:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            val1 = df1[axis].values
            val2 = df2[axis].values

            is_angle = axis in ["roll", "pitch", "yaw"]
            error, mae, max_err, rmse = calculate_error_metrics(val1, val2, is_angle)

            # Data plots
            ax1.plot(time, val1, label=f'{axis} (Real-UR10e)', linewidth=2)
            ax1.plot(time, val2, label=f'{axis} (FMU-UR10e)', linestyle='--', linewidth=2)

            # Error plot
            ax2 = ax1.twinx()
            ax2.plot(time, error, label='Error', color='black', linewidth=1.5, alpha=0.6)

            # Titles and labels
            ax1.set_title(f"{category_title} - {axis} Comparison\n"
                          f"MAE: {mae:.4f} | Max Error: {max_err:.4f} | RMSE: {rmse:.4f}", fontsize=14)
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel(f"{axis} Value")
            ax2.set_ylabel("Error (Absolute)")

            # Legends
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            # Highlight max error
            if max_err > 0.01:
                max_idx = np.argmax(error)
                ax1.axvline(time[max_idx], color='red', linestyle=':', alpha=0.5)
                ax1.text(time[max_idx], np.mean(val1),
                         f'Max Error @ {time[max_idx]}s',
                         rotation=90, verticalalignment='bottom', fontsize=10)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

print("PDF successfully created with angle normalization: robot_comparison_analysis.pdf")
