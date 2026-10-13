import numpy as np
import matplotlib.pyplot as plt

R = 2000.0          
C = 0.001           
tau = R * C         

V_initial = 12.0    
h = 0.1            
t_end = 10.0        

time = np.arange(0, t_end + h, h)
voltage_euler = np.zeros_like(time)
voltage_euler[0] = V_initial

for i in range(len(time) - 1):
    dV_dt = -voltage_euler[i] / tau
    voltage_euler[i + 1] = voltage_euler[i] + h * dV_dt

voltage_exact = V_initial * np.exp(-time / tau)
final_error = abs(voltage_euler[-1] - voltage_exact[-1])

print(f"Capacitor Voltage at 10s (Euler): {voltage_euler[-1]:.4f} V")
print(f"Capacitor Voltage at 10s (Exact): {voltage_exact[-1]:.4f} V")
print(f"Absolute Truncation Error:        {final_error:.4f} V")

plt.figure(figsize=(8, 4))
plt.plot(time, voltage_euler, label=f"Euler Method (h={h}s)", color="crimson")
plt.plot(time, voltage_exact, label="Analytical Exact Solution", linestyle="--", color="black")
plt.xlabel("Time (seconds)")
plt.ylabel("Capacitor Voltage (V)")
plt.title("Capacitor Discharge Transient Analysis (Euler Method)")
plt.legend()
plt.grid(True)
plt.savefig("adaptations/task_17_capacitor_discharge.png")
plt.show()