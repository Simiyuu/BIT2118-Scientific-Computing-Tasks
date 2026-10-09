import numpy as np

transaction_amount = 10.10
fee_rate = 0.0005
transactions_count = 1_000_000

exact_total_fees = transactions_count * 0.00505

naive_total = 0.0
for _ in range(transactions_count):
    naive_total += (transaction_amount * fee_rate)

vectorized_total = np.sum(np.full(transactions_count, transaction_amount * fee_rate))

abs_err = abs(exact_total_fees - naive_total)
rel_err = abs_err / exact_total_fees

print(f"Expected theoretical ledger total: ${exact_total_fees:.6f}")
print(f"Naive incremental sum total:       ${naive_total:.6f}")
print(f"NumPy vectorized sum total:         ${vectorized_total:.6f}")
print(f"Absolute drift error:              ${abs_err:.8f}")
print(f"Relative drift percentage:         {rel_err:.8%}")
print("Is naive sum equal to expected?", naive_total == exact_total_fees)
print("Is naive sum close within tolerance?", np.isclose(naive_total, exact_total_fees, atol=1e-5))