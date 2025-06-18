import pandas as pd

# Giriş dosyasının yolu
input_excel_path = r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_joint_data_processed.xlsx"

# Çıkış dosyasının yolu
output_csv_path = r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\fmpy_inputs.csv"

# Excel dosyasını oku
df = pd.read_excel(input_excel_path, engine='openpyxl')

# CSV olarak kaydet (virgül ayraçlı, başlık dahil)
df.to_csv(output_csv_path, index=False)
