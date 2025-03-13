import numpy as np
import matplotlib.pyplot as plt
from model import create_fmu_instance

# Simulation parameters
DT = 0.1                  # zaman adım büyüklüğü (s)
T_total = 10.0            # toplam simülasyon süresi (s)
time = np.arange(0, T_total + DT, DT)  # zaman vektörü

# FMU instance oluşturuluyor
fmu = create_fmu_instance()

# Başlangıç koşulları
fmu.x_des = 0.5
fmu.y_des = 0.0
fmu.z_des = 0.0
fmu.roll_des = 0.0
fmu.pitch_des = 0.0
fmu.yaw_des = 0.0
fmu.kp = 7.0
fmu.kd = 3.0

# transient response ölçümleri için listeler
time_data = []
ee_x_data = []      # end-effector x pozisyonu

# Simülasyon döngüsü:
for t in time:
    # t = 2.0 s'de step giriş: x_des değerini 0.5'ten 1.0 m'ye değiştirelim.
    if t >= 2.0:
        fmu.x_des = 1.0
    fmu.do_step(t, DT, False)
    time_data.append(t)
    ee_pose = fmu.ee_pose  # [x, y, z, roll, pitch, yaw]
    ee_x_data.append(ee_pose[0])

# Transient Response Metriklerini Hesaplama
steady_state_value = np.mean(ee_x_data[-10:])  # son 10 değerin ortalaması
peak_value = np.max(ee_x_data)
peak_time = time_data[np.argmax(ee_x_data)]

# Rise time: %10 ve %90 değerlerini bul
lower_bound = 0.1 * steady_state_value
upper_bound = 0.9 * steady_state_value
rise_start_idx = next((i for i, val in enumerate(ee_x_data) if val >= lower_bound), None)
rise_end_idx = next((i for i, val in enumerate(ee_x_data) if val >= upper_bound), None)
rise_time = time_data[rise_end_idx] - time_data[rise_start_idx] if rise_start_idx is not None and rise_end_idx is not None else None

# Overshoot: (peak_value - steady_state_value) / steady_state_value * 100
overshoot = ((peak_value - steady_state_value) / steady_state_value * 100) if steady_state_value != 0 else None

# Steady-state error: |steady_state_value - x_des|
steady_state_error = abs(steady_state_value - fmu.x_des)

# Grafik oluşturma
plt.figure(figsize=(10, 6))
plt.plot(time_data, ee_x_data, label='End-Effector x Position')
plt.xlabel('Time (s)')
plt.ylabel('x Position (m)')
plt.title('Transient Response Analysis: Step Input at t = 2 s')
plt.grid(True)

# Anotasyonlar
if rise_time is not None:
    plt.annotate(f"Rise Time: {rise_time:.2f} s", xy=(time_data[rise_end_idx], ee_x_data[rise_end_idx]),
                 xytext=(time_data[rise_end_idx]+0.5, ee_x_data[rise_end_idx]+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate(f"Peak Time: {peak_time:.2f} s", xy=(peak_time, peak_value),
             xytext=(peak_time, peak_value+0.05),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate(f"Overshoot: {overshoot:.1f}%", xy=(peak_time, peak_value),
             xytext=(peak_time+0.5, peak_value+0.05),
             arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate(f"Steady-State Error: {steady_state_error:.3f} m", xy=(time_data[-1], steady_state_value),
             xytext=(time_data[-1]-2, steady_state_value-0.05),
             arrowprops=dict(facecolor='blue', shrink=0.05))

plt.legend()
plt.show()
