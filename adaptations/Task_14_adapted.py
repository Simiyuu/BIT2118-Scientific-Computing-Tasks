import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def discharge_model(t, a, b, c):
    return a * np.exp(-b * t) + c

hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
voltage_measured = np.array([4.20, 4.05, 3.92, 3.81, 3.72, 3.64, 3.55, 3.42, 3.20])

initial_params = [1.0, 0.05, 3.2]
params, cov = curve_fit(discharge_model, hours, voltage_measured, p0=initial_params)
a, b, c = params

fitted_v = discharge_model(hours, a, b, c)
rmse = np.sqrt(np.mean((voltage_measured - fitted_v) ** 2))

print(f"Fitted Model: V(t) = {a:.3f} * exp(-{b:.4f} * t) + {c:.3f}")
print(f"Fit RMSE: {rmse:.4f} V")
print(f"Predicted battery voltage at 24 hours: {discharge_model(24, a, b, c):.2f} V")

t_eval = np.linspace(0, 24, 200)
plt.figure(figsize=(8, 4))
plt.scatter(hours, voltage_measured, color="blue", label="Sensor Log")
plt.plot(t_eval, discharge_model(t_eval, a, b, c), color="red", label="Nonlinear Least Squares Fit")
plt.axhline(3.0, color="black", linestyle="--", label="Cutoff Threshold (3.0V)")
plt.title("IoT Battery Discharge Profile Fit")
plt.xlabel("Time (Hours)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.savefig("adaptations/task_14_battery_curve.png")
plt.show()