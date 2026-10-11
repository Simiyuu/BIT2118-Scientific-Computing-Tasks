import numpy as np
from scipy.optimize import root_scalar

V_source = 5.0       
R = 1000.0           
I_s = 1e-12          
V_t = 0.026          
n = 1.0              

def diode_residual(V_d):
    return (V_source - V_d) / R - I_s * (np.exp(V_d / (n * V_t)) - 1.0)

def d_diode_residual(V_d):
    return -1.0 / R - (I_s / (n * V_t)) * np.exp(V_d / (n * V_t))

sol_newton = root_scalar(
    diode_residual,
    fprime=d_diode_residual,
    x0=0.6,
    method="newton",
    xtol=1e-9
)

sol_secant = root_scalar(
    diode_residual,
    x0=0.4,
    x1=0.8,
    method="secant",
    xtol=1e-9
)

current_mA = ((V_source - sol_newton.root) / R) * 1000.0

print(f"Diode Voltage (Newton-Raphson): {sol_newton.root:.6f} V")
print(f"Diode Voltage (Secant Method):   {sol_secant.root:.6f} V")
print(f"Circuit Forward Current:        {current_mA:.3f} mA")
print(f"Residual Check:                 {diode_residual(sol_newton.root):.2e}")