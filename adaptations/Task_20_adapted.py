import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(77)
N_trials = 50_000

annual_breach_counts = rng.poisson(lam=3.0, size=N_trials)
annual_losses = np.zeros(N_trials)

for i in range(N_trials):
    n_incidents = annual_breach_counts[i]
    if n_incidents > 0:
        losses = rng.lognormal(mean=11.5, sigma=1.2, size=n_incidents)
        annual_losses[i] = np.sum(losses)

mean_loss = np.mean(annual_losses)
p95_var = np.percentile(annual_losses, 95)
prob_exceed_2m = np.mean(annual_losses > 2_000_000)

print(f"Mean Expected Annual Loss:       ${mean_loss:,.2f}")
print(f"95% Value at Risk (VaR):         ${p95_var:,.2f}")
print(f"Probability of Loss > $2 Million: {prob_exceed_2m:.2%}")

plt.figure(figsize=(8, 4))
plt.hist(annual_losses / 1e6, bins=60, color="navy", edgecolor="white", density=True)
plt.axvline(p95_var / 1e6, color="red", linestyle="--", label=f"95% VaR: ${p95_var/1e6:.2f}M")
plt.title("Monte Carlo Annual Cyber Loss Distribution")
plt.xlabel("Annual Loss ($ Millions)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.savefig("adaptations/task_20_cyber_risk.png")
plt.show()