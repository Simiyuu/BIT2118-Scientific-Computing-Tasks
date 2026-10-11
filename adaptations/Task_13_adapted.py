import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt


t_logged = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0])
speed_logged = np.array([0.0, 25.0, 50.0, 65.0, 70.0, 60.0, 0.0])

linear_telemetry = interp1d(t_logged, speed_logged, kind="linear")
spline_telemetry = CubicSpline(t_logged, speed_logged)

t_fine = np.linspace(0.0, 60.0, 300)
print(f"Interpolated speed at t=15s (Linear): {float(linear_telemetry(15.0)):.2f} km/h")
print(f"Interpolated speed at t=15s (Spline): {float(spline_telemetry(15.0)):.2f} km/h")

plt.figure(figsize=(8, 4))
plt.scatter(t_logged, speed_logged, color="black", zorder=3, label="Logged GPS Samples")
plt.plot(t_fine, linear_telemetry(t_fine), label="Linear Interpolation", linestyle=":")
plt.plot(t_fine, spline_telemetry(t_fine), label="Cubic Spline Reconstruction", color="green")
plt.title("Vehicle Speed Telemetry Reconstruction")
plt.xlabel("Time (seconds)")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(True)
plt.savefig("adaptations/task_13_telemetry_spline.png")
plt.show()