import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import get_all_joint_positions

# 1. Simülasyon verisini oku
df = pd.read_csv("ur10e_FK_out.csv")
joint_trajectories = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].values

# 2. Her adımda tüm joint pozisyonlarını hesapla
robot_joint_paths = [get_all_joint_positions(q) for q in joint_trajectories]

# 3. Matplotlib figürünü kur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(0.0, 1.5)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("UR10e Robot - Joint Based Visualization")

# 4. Başlangıç çizim objeleri
link_line, = ax.plot([], [], [], 'o-', lw=4, c='blue', label='Robot Links')
ee_trace, = ax.plot([], [], [], 'r--', lw=1, label='End-Effector Path')
trajectory = []

# 5. Her frame'de güncelleme fonksiyonu
def update(frame):
    joints = robot_joint_paths[frame]  # 7x3
    ee_pos = joints[-1]
    trajectory.append(ee_pos)

    link_line.set_data(joints[:, 0], joints[:, 1])
    link_line.set_3d_properties(joints[:, 2])

    traj_arr = np.array(trajectory)
    ee_trace.set_data(traj_arr[:, 0], traj_arr[:, 1])
    ee_trace.set_3d_properties(traj_arr[:, 2])

    return link_line, ee_trace

# 6. Animasyonu başlat
ani = FuncAnimation(fig, update, frames=len(robot_joint_paths), interval=30)
plt.legend()
plt.tight_layout()
plt.show()
