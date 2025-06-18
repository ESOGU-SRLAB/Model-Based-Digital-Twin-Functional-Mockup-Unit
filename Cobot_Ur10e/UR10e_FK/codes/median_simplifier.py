import pandas as pd

def simplify_to_median(input_file, output_file):
    # Load the Excel file
    df = pd.read_excel(input_file)

    # Filter only the specified columns
    df = df[["time", "q1", "q2", "q3", "q4", "q5", "q6", "x", "y", "z", "roll", "pitch", "yaw", "qx", "qy", "qz", "qw"]]

    # Drop rows with non-finite values in the 'time' column
    df = df[pd.to_numeric(df['time'], errors='coerce').notna()]

    df['second'] = df['time'].astype(int)

    # Group by each second and take the row corresponding to the median value in the 'time' column
    median_rows = df.groupby('second', group_keys=False).apply(
        lambda group: group.loc[group['time'].idxmin() + len(group) // 2]
    ).reset_index(drop=True)

    # Drop the 'second' column and save only the specified columns
    median_rows = median_rows[["time", "q1", "q2", "q3", "q4", "q5", "q6", "x", "y", "z", "roll", "pitch", "yaw", "qx", "qy", "qz", "qw"]]

    # Save the result based on the file extension
    if output_file.endswith('.csv'):
        median_rows.to_csv(output_file, index=False)
    elif output_file.endswith('.xlsx'):
        median_rows.to_excel(output_file, index=False)
    else:
        raise ValueError("Unsupported file format. Use '.csv' or '.xlsx'.")

# Example usage
input_file = r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_joint_data_processed.xlsx"  # Replace with your input file path
output_file = r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_joint_data_processed_v2.xlsx"  # Replace with your desired output file path
simplify_to_median(input_file, output_file)

