import numpy as np
import matplotlib.pyplot as plt

T_ambient = 25.0
T_initial = 90.0
k = 0.12
h = 1.0  
end_time = 30.0

def cooling_rate(T):
    return -k * (T - T_ambient)

time = np.arange(0, end_time + h, h)
temperature = np.zeros_like(time)
temperature[0] = T_initial

for i in range(len(time) - 1):
    temperature[i + 1] = temperature[i] + h * cooling_rate(temperature[i])


exact = T_ambient + (T_initial - T_ambient) * np.exp(-k * time)

print(f"Euler temperature after 30 min: {temperature[-1]:.2f} C")
print(f"Exact temperature after 30 min: {exact[-1]:.2f} C")
print(f"Final absolute error:          {abs(temperature[-1] - exact[-1]):.4f} C")

plt.plot(time, temperature, marker="o", label="Euler (h=1.0)")
plt.plot(time, exact, label="Exact solution", linestyle="--")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (C)")
plt.title("Cooling of a Hot Object (Euler Method)")
plt.legend()
plt.grid(True)
plt.savefig("starter/task_17_euler_cooling.png")
plt.show()