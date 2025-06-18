import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === 1. LOAD DATA ===
df1 = pd.read_excel(r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_joint_data_processed.xlsx")
df2 = pd.read_excel(r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\ur10e_out.xlsx")
time = df1["time"]

# === 2. FUNCTION TO CALCULATE ERROR METRICS ===
def calculate_error_metrics(true_vals, pred_vals):
    error = np.abs(true_vals - pred_vals)
    mae = np.mean(error)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    return error, mae, max_error, rmse

# === 3. DEFINE CATEGORIES ===
categories = {
    "Position (x, y, z)": ["x", "y", "z"],
    "Euler Angles (roll, pitch, yaw)": ["roll", "pitch", "yaw"],
    "Quaternions (qx, qy, qz, qw)": ["qx", "qy", "qz", "qw"]
}

# === 4. PDF GRAPH OUTPUT ===
with PdfPages(r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_comparison_analysis.pdf") as pdf:
    plt.style.use("ggplot")
    for category_title, axes in categories.items():
        for axis in axes:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            val1 = df1[axis].values
            val2 = df2[axis].values

            # Errors and metrics
            error, mae, max_err, rmse = calculate_error_metrics(val1, val2)

            # Real and target data
            ax1.plot(time, val1, label=f'{axis} (Real-UR10e)', linewidth=2)
            ax1.plot(time, val2, label=f'{axis} (FMU-UR10e)', linestyle='--', linewidth=2)

            # Error curve
            ax2 = ax1.twinx()
            ax2.plot(time, error, label='Error', color='black', linewidth=1.5, alpha=0.6)

            # Title and labels
            ax1.set_title(f"{category_title} - {axis} Comparison\n"
                          f"MAE: {mae:.4f} | Max Error: {max_err:.4f} | RMSE: {rmse:.4f}", fontsize=14)
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel(f"{axis} Value")
            ax2.set_ylabel("Error (Absolute)")

            # Legends
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            # Mark maximum error
            if max_err > 0.01:
                max_err_idx = np.argmax(error)
                ax1.axvline(time[max_err_idx], color='red', linestyle=':', alpha=0.5)
                ax1.text(time[max_err_idx], np.mean(val1),
                         f'Max Error @ {time[max_err_idx]}s',
                         rotation=90, verticalalignment='bottom', fontsize=10)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

print("PDF successfully created: robot_comparison_analysis.pdf")
