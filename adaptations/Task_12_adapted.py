import numpy as np
from scipy.optimize import root

def trilateration_residuals(pos):
    x, y = pos
    r1_sq = x**2 + y**2 - 25.0
    r2_sq = (x - 4.0)**2 + (y - 1.0)**2 - 13.0
    return [r1_sq, r2_sq]

guesses = [[3.0, 4.0], [4.0, -2.0]]

print("Rogue Transmitter Multilateration Solutions:")
for i, g in enumerate(guesses, 1):
    res = root(trilateration_residuals, g)
    print(f"Guess {i}: {g} -> Emitter Position: x = {res.x[0]:.3f} km, y = {res.x[1]:.3f} km")
    print(f"  Converged: {res.success}, Residual Norm: {np.linalg.norm(res.fun):.2e}")