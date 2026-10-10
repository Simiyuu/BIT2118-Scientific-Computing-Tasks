import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

def financial_balance(N):
    return (15.0 * N) - (12000.0 + 8.0 * N + 4000.0 * np.log(np.maximum(N, 1.0)))

def g_financial(N):
    return (12000.0 + 4000.0 * np.log(np.maximum(N, 1.0))) / 7.0

bisect_res = root_scalar(
    financial_balance,
    bracket=[1000, 8000],
    method="bisect",
    xtol=1e-5
)

N_est = 2000.0
fp_history = [N_est]
for _ in range(150):
    N_next = g_financial(N_est)
    fp_history.append(N_next)
    if abs(N_next - N_est) < 1e-5:
        break
    N_est = N_next

print(f"Bisection Break-Even Subscribers:   {bisect_res.root:.2f}")
print(f"Fixed-Point Break-Even Subscribers: {fp_history[-1]:.2f}")
print(f"Fixed-Point Iteration Count:        {len(fp_history) - 1}")


users = np.linspace(1000, 8000, 400)
plt.figure(figsize=(8, 4))
plt.plot(users, financial_balance(users), label="Net Profit / Loss Curve", color="purple")
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.axvline(bisect_res.root, color="green", linestyle=":", label=f"Break-even: {bisect_res.root:.0f} users")
plt.xlabel("Active Paid Subscribers (N)")
plt.ylabel("Net Balance ($)")
plt.title("SaaS Break-Even Point Root Finding")
plt.legend()
plt.grid(True)
plt.savefig("adaptations/task_10_breakeven.png")
plt.show()