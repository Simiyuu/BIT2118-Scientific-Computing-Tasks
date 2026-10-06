import numpy as np

bandwidth_mbps = np.array([
    42.0, 38.5, 30.2, 25.0, 28.4, 45.1,
    95.6, 160.4, 220.0, 275.5, 305.2, 312.0,
    310.5, 298.0, 285.4, 270.1, 250.0, 225.8,
    205.2, 180.0, 150.3, 110.6, 75.0, 52.4
])

mean_bandwidth = np.mean(bandwidth_mbps)
min_bandwidth = np.min(bandwidth_mbps)
peak_bandwidth = np.max(bandwidth_mbps)

hourly_gigabytes = bandwidth_mbps * (3600 / 8000)
total_daily_gb = np.sum(hourly_gigabytes)

print(f"Daily Average Bandwidth: {mean_bandwidth:.2f} Mbps")
print(f"Minimum (Trough) Bandwidth: {min_bandwidth:.2f} Mbps")
print(f"Peak (Maximum) Bandwidth: {peak_bandwidth:.2f} Mbps")
print(f"Total Daily Transferred Data: {total_daily_gb:.2f} GB")
print("First 6 hours of transferred data (GB):", np.round(hourly_gigabytes[:6], 2))