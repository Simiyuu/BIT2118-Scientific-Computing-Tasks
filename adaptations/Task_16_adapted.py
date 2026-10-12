import numpy as np
from scipy.integrate import simpson

time_hours = np.arange(0, 13, 1, dtype=float)
time_seconds = time_hours * 3600.0

throughput_mbps = np.array([
    45.0, 52.0, 78.0, 120.0, 210.0, 340.0,
    380.0, 360.0, 290.0, 210.0, 140.0, 85.0, 60.0
])

total_mbits_trapz = np.trapezoid(throughput_mbps, time_seconds)
total_mbits_simps = simpson(throughput_mbps, x=time_seconds)

total_gb_trapz = total_mbits_trapz / 8000.0
total_gb_simps = total_mbits_simps / 8000.0

print(f"Total Traffic (Trapezoidal Rule): {total_gb_trapz:.2f} GB")
print(f"Total Traffic (Simpson's Rule):   {total_gb_simps:.2f} GB")
print(f"Integration Discrepancy:          {abs(total_gb_simps - total_gb_trapz):.3f} GB")