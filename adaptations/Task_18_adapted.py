import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N_total = 1000.0    
beta = 0.45          
gamma = 0.10        
I0 = 5.0             

def malware_spread(t, y):
    I = y[0]
    dI_dt = beta * I * (N_total - I) / N_total - gamma * I
    return [dI_dt]

t_span = (0, 30)     
t_eval = np.linspace(0, 30, 300)

sol = solve_ivp(malware_spread, t_span, [I0], t_eval=t_eval, method="RK45")

theoretical_equilibrium = N_total * (1.0 - gamma / beta)

print("Simulation Status:", sol.success)
print(f"Infected machines at 5 hours:  {sol.y[0, 50]:.1f}")
print(f"Infected machines at 30 hours: {sol.y[0, -1]:.1f}")
print(f"Theoretical steady-state limit: {theoretical_equilibrium:.1f}")

plt.figure(figsize=(8, 4))
plt.plot(sol.t, sol.y[0], color="crimson", linewidth=2, label="Infected Hosts I(t)")
plt.axhline(theoretical_equilibrium, color="black", linestyle="--", label=f"Equilibrium ({theoretical_equilibrium:.0f} hosts)")
plt.title("Malware Propagation Dynamics (SIS Model with solve_ivp)")
plt.xlabel("Time (Hours)")
plt.ylabel("Infected Endpoints")
plt.grid(True)
plt.legend()
plt.savefig("adaptations/task_18_malware_spread.png")
plt.show()