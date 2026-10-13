import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
capacity = 1000.0
mean_demand = 850.0
std_demand = 120.0
N = 100_000

demand = rng.normal(mean_demand, std_demand, N)
overloaded = demand > capacity
overload_probability = overloaded.mean()

print(f"Estimated overload probability: {overload_probability:.4%}")
print(f"Number of overload cases:        {overloaded.sum()} out of {N}")

running_probability = np.cumsum(overloaded) / np.arange(1, N + 1)
indices = np.unique(
    np.logspace(1, np.log10(N), 200).astype(int)
) - 1

plt.semilogx(indices + 1, running_probability[indices], color="crimson")
plt.axhline(overload_probability, color="black", linestyle="--", label=f"Final P: {overload_probability:.4%}")
plt.xlabel("Number of simulations")
plt.ylabel("Estimated overload probability")
plt.title("Monte Carlo Estimate of Server-Overload Risk")
plt.grid(True)
plt.legend()
plt.savefig("starter/task_20_monte_carlo.png")
plt.show()