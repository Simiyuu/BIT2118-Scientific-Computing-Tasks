import numpy as np
import matplotlib.pyplot as plt

time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
position = np.array([0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])

speed = np.zeros_like(position)

speed[0] = (position[1] - position[0]) / (time[1] - time[0])

for i in range(1, len(time) - 1):
    speed[i] = (position[i + 1] - position[i - 1]) / (time[i + 1] - time[i - 1])

speed[-1] = (position[-1] - position[-2]) / (time[-1] - time[-2])

for t, x, v in zip(time, position, speed):
    print(f"t={t:.0f}s, position={x:.1f}m, estimated speed = {v:.2f} m/s")

plt.plot(time, speed, marker="o", color="blue", label="Speed (diff)")
plt.xlabel("Time (s)")
plt.ylabel("Estimated speed (m/s)")
plt.title("Speed Estimated from GPS Position Data")
plt.grid(True)
plt.legend()
plt.savefig("starter/task_15_numerical_differentiation.png")
plt.show()