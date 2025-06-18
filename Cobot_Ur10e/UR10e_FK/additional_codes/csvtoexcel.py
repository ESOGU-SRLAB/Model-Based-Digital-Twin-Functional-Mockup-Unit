import pandas as pd

input_file = r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\ur10e_FK_outs.csv"
output_file = r"c:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\ur10e_outs.xlsx"

# Read the CSV file using a comma as the delimiter
df = pd.read_csv(input_file, delimiter=",")

# Write the DataFrame to an Excel file without the index column
df.to_excel(output_file, index=False)

print(f"Conversion complete: {output_file}")