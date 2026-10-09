import numpy as np

patrol_waypoints = np.array([
    [10.0, 10.0],
    [50.0, 10.0],
    [50.0, 40.0],
    [10.0, 40.0]
])

theta = np.deg2rad(30)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

S = np.array([
    [1.2, 0.0],
    [0.0, 1.5]
])

composite_T = R @ S

updated_waypoints = patrol_waypoints @ composite_T.T

area_scale_factor = np.linalg.det(composite_T)

print("Original Patrol Waypoints (m):\n", patrol_waypoints)
print("\nComposite Transformation Matrix:\n", np.round(composite_T, 4))
print("\nUpdated Drone Patrol Waypoints (m):\n", np.round(updated_waypoints, 2))
print(f"\nPerimeter Area Scaling Factor (det): {area_scale_factor:.2f}")