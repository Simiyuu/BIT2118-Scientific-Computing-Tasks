import numpy as np
import timeit

rng = np.random.default_rng(101)
packet_sizes_bytes = rng.integers(64, 1518, size=1_000_000)

min_val = 64.0
max_val = 1518.0

def iterative_normalizer():
    normalized = []
    for size in packet_sizes_bytes:
        scaled = (size - min_val) / (max_val - min_val)
        normalized.append(scaled)
    return normalized

def vectorized_normalizer():
    return (packet_sizes_bytes - min_val) / (max_val - min_val)

loop_exec = timeit.timeit(iterative_normalizer, number=1)
vec_exec = timeit.timeit(vectorized_normalizer, number=5) / 5
scaled_packets = vectorized_normalizer()

print("First five normalized packet values:", np.round(scaled_packets[:5], 4))
print(f"Python standard loop time: {loop_exec:.4f} s")
print(f"Vectorized NumPy time:      {vec_exec:.4f} s")
print(f"Observed performance boost: {loop_exec / vec_exec:.1f}x")