import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import forward_kinematics
import os

csv_path = "fmpy_inputs.csv"
q_columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
last_len = 0
positions = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([], [], [], c='red', s=50)
line, = ax.plot([], [], [], 'b-', lw=2)
trajectory = []

def update(_):
    global last_len, positions

    if not os.path.exists(csv_path):
        return scatter, line

    df = pd.read_csv(csv_path)

    if len(df) > last_len:
        new_rows = df.iloc[last_len:]
        for _, row in new_rows.iterrows():
            q = row[q_columns].values
            pos = forward_kinematics(q)[:3]
            positions.append(pos)
        last_len = len(df)

    if not positions:
        return scatter, line

    pt = positions[-1]
    trajectory.append(pt)
    traj = np.array(trajectory)

    scatter._offsets3d = ([pt[0]], [pt[1]], [pt[2]])
    line.set_data(traj[:, 0], traj[:, 1])
    line.set_3d_properties(traj[:, 2])
    return scatter, line

ani = FuncAnimation(fig, update, interval=100)
plt.tight_layout()
plt.show()
