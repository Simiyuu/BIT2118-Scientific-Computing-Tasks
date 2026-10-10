import numpy as np
import matplotlib.pyplot as plt

A_pipe = np.array([
    [ 2.5, -1.0,  0.0,  0.0,  0.0],
    [-1.0,  2.5, -1.0,  0.0,  0.0],
    [ 0.0, -1.0,  2.5, -1.0,  0.0],
    [ 0.0,  0.0, -1.0,  2.5, -1.0],
    [ 0.0,  0.0,  0.0, -1.0,  2.5]
])
b_pipe = np.array([80.0, 0.0, 0.0, 0.0, 15.0])

def solve_gauss_seidel(A, b, tol=1e-7, max_iter=300):
    x = np.zeros_like(b)
    history = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ old[i+1:]) / A[i, i]
        diff = np.linalg.norm(x - old, ord=np.inf)
        history.append(diff)
        if diff < tol:
            return x, history
    return x, history

pressure, err_history = solve_gauss_seidel(A_pipe, b_pipe)
exact_pressure = np.linalg.solve(A_pipe, b_pipe)

print("Estimated Junction Pressures (PSI):", np.round(pressure, 2))
print("Direct Exact Pressures (PSI):       ", np.round(exact_pressure, 2))
print(f"Converged in {len(err_history)} iterations.")

plt.figure(figsize=(8, 4))
plt.semilogy(err_history, marker='o', color='teal', label='Gauss-Seidel Error')
plt.title("Pipeline Hydraulic Pressure Convergence")
plt.xlabel("Iteration")
plt.ylabel("Max Pressure Change (PSI)")
plt.grid(True)
plt.legend()
plt.savefig("adaptations/task_08_pipeline_convergence.png")
plt.show()