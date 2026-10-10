import numpy as np

A_cluster = np.array([
    [0., 1., 0., 1., 0.],  
    [1., 0., 1., 1., 1.],  
    [0., 1., 0., 1., 0.],  
    [1., 1., 1., 0., 0.],  
    [0., 1., 0., 0., 0.]   
])

eigenvalues, eigenvectors = np.linalg.eig(A_cluster)

idx = np.argmax(np.real(eigenvalues))
dominant_eigval = np.real(eigenvalues[idx])
eigenvector_centrality = np.abs(np.real(eigenvectors[:, idx]))
centrality_scores = eigenvector_centrality / np.sum(eigenvector_centrality)

service_names = [
    "Auth Service",
    "API Gateway",
    "Payment Service",
    "User DB",
    "Notification Svc"
]

print(f"Dominant Eigenvalue (Spectral Radius): {dominant_eigval:.4f}\n")
print("Microservice Criticality / Attack Surface Centrality:")
for name, score in zip(service_names, centrality_scores):
    print(f"  {name:18}: {score * 100:.2f}%")

v = np.real(eigenvectors[:, idx])
recon_error = np.linalg.norm(A_cluster @ v - dominant_eigval * v)
print(f"\nVerification Residual Norm: {recon_error:.2e}")