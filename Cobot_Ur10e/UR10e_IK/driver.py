from pyfmi import load_fmu
import numpy as np

# FMU’yu yükle
fmu = load_fmu("UR10e_InverseKinematics.fmu")

# Başlangıç zamanı ayarla
fmu.setup_experiment(start_time=0.0)
fmu.enter_initialization_mode()
fmu.exit_initialization_mode()

# "Elbow-up" pozuna nearest devam:
# Örneğin
x_des, y_des, z_des = 0.3, 0.1, 0.5
roll_des, pitch_des, yaw_des = 0.0, 0.0, 0.0

fmu.set([0,1,2,3,4,5], [x_des, y_des, z_des, roll_des, pitch_des, yaw_des])
fmu.do_step(current_t=0.0, step_size=1e-3) 

# Çıktıları al:
q_vals, status = fmu.get([6,7,8,9,10,11])
print("Found IK solution:", q_vals)

fmu.terminate()
