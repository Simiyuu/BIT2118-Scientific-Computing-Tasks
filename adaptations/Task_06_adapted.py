import numpy as np

A = np.array([
    [8.0, 4.0],   
    [16.0, 32.0]   
])
b = np.array([88.0, 224.0])

vm_solution = np.linalg.solve(A, b)
residual = b - A @ vm_solution

print(f"Active IDS Inspection VMs (x):  {vm_solution[0]:.0f}")
print(f"Active SIEM Aggregator VMs (y): {vm_solution[1]:.0f}")
print("Verification A @ x:", A @ vm_solution)
print("Residual vector:", residual)
print(f"Residual norm: {np.linalg.norm(residual):.2e}")