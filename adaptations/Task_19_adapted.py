import numpy as np
from scipy.optimize import minimize

def infrastructure_cost(z):
    x1, x2 = z
    return 0.20 * x1 + 0.08 * x2

def throughput_constraint(z):
    x1, x2 = z
    return (50.0 * x1 + 35.0 * x2) - 1200.0

constraints = [
    {"type": "ineq", "fun": throughput_constraint}
]

bounds = [(0, None), (0, 12)]

res = minimize(
    infrastructure_cost,
    x0=[20.0, 5.0],
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

x1_opt, x2_opt = res.x
print("Optimization Status:", res.success)
print(f"Optimal On-Demand Instances (x1): {x1_opt:.2f}")
print(f"Optimal Spot Instances (x2):      {x2_opt:.2f}")
print(f"Total Throughput Provided:       {50*x1_opt + 35*x2_opt:.1f} req/s")
print(f"Minimum Hourly Hosting Cost:     ${res.fun:.2f}/hr")