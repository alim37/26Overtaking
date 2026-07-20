#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = os.path.expanduser("~/aaron_10laps.csv") 
out_dir  = os.path.expanduser("~/RESEARCH")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

t = df["t"].to_numpy()
x = df["x"].to_numpy()
y = df["y"].to_numpy()
vx = df["vx"].to_numpy()
vy = df["vy"].to_numpy()
yaw_rate = df["yaw_rate"].to_numpy()

t_sec = t - t[0]

plt.figure()
plt.plot(x, y, label="Trajectory")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trajectory (x-y)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_dir, "trajectory.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(t_sec, vx, label="$v_x$")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Longitudinal Velocity")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_dir, "vx.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(t_sec, vy, label="$v_y$")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Lateral Velocity")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_dir, "vy.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(t_sec, yaw_rate, label=r"$\dot{\psi}$")
plt.xlabel("Time [s]")
plt.ylabel("Yaw Rate [rad/s]")
plt.title("Yaw Rate")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_dir, "yaw_rate.png"), dpi=300, bbox_inches="tight")
plt.close()
