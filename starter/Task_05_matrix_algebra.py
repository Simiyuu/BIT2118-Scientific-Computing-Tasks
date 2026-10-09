import numpy as np

points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

angle = np.deg2rad(45)
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])

scale = np.array([
    [2.0, 0.0],
    [0.0, 1.5]
])

transformation = rotation @ scale

transformed_points = points @ transformation.T

print("Original points: \n", points)
print("Transformation matrix:\n", transformation)
print("Transformed points: \n", transformed_points)
print("Determinant:", np.linalg.det(transformation))