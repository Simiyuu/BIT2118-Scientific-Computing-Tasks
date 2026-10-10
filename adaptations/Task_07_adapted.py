import numpy as np
from scipy.linalg import lu_factor, lu_solve

A_grid = np.array([
    [5.0, -2.0,  0.0],
    [-2.0, 6.0, -3.0],
    [0.0, -3.0,  4.0]
])

current_injections = [
    np.array([24.0, 18.0, 10.0]),
    np.array([30.0, 22.0, 15.0]),
    np.array([12.0,  8.0,  5.0])
]

lu_factors, pivot_indices = lu_factor(A_grid)

regimes = ["Morning Peak", "Afternoon", "Night Off-Peak"]
for name, I_vec in zip(regimes, current_injections):
    voltages = lu_solve((lu_factors, pivot_indices), I_vec)
    res_norm = np.linalg.norm(I_vec - A_grid @ voltages)
    print(f"Regime: {name}")
    print(f"  Bus Voltages (V): {np.round(voltages, 4)}")
    print(f"  Residual Norm:    {res_norm:.2e}")