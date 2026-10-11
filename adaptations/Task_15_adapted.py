import numpy as np
import matplotlib.pyplot as plt

time_s = np.array([0, 2, 4, 6, 8, 10], dtype=float)
cumulative_requests = np.array([1200, 1340, 1620, 2180, 3100, 4500], dtype=float)

request_rate = np.zeros_like(cumulative_requests)

request_rate[0] = (cumulative_requests[1] - cumulative_requests[0]) / (time_s[1] - time_s[0])

for i in range(1, len(time_s) - 1):
    dt = time_s[i + 1] - time_s[i - 1]
    request_rate[i] = (cumulative_requests[i + 1] - cumulative_requests[i - 1]) / dt

request_rate[-1] = (cumulative_requests[-1] - cumulative_requests[-2]) / (time_s[-1] - time_s[-2])

print("Traffic Rate Analysis (DDoS Ramp Detection):")
for t, total, rate in zip(time_s, cumulative_requests, request_rate):
    print(f"  t = {t:2.0f}s | Cumulative: {total:5.0f} req | Influx Rate: {rate:6.1f} req/s")

plt.figure(figsize=(8, 4))
plt.plot(time_s, request_rate, marker="s", color="darkred", label="Rate of Change d(Req)/dt")
plt.axhline(500.0, color="orange", linestyle="--", label="DDoS Alert Threshold (500 req/s)")
plt.title("Instantaneous Request Rate (Numerical Differentiation)")
plt.xlabel("Time (seconds)")
plt.ylabel("Traffic Velocity (Requests/sec)")
plt.grid(True)
plt.legend()
plt.savefig("adaptations/task_15_ddos_rate.png")
plt.show()