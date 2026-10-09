# BIT2118-Scientific-Computing-Tasks

* **Student Name:** [Simiyu Chandler Matere]
* **Registration Number:** [BSCCS/2024/58571]
* **Course:** BIT 2118 - Scientific Computing
* **Coursework Component:** Group Collaboration & Individual Practical Portfolio

---

## Prerequisite / Task 01: Data Science Essentials with Python

Before tackling the numerical computation modules, I completed the self-paced foundation course **Data Science Essentials with Python**. This course provided the core refresher on Python data structures, vectorized operations with NumPy, and clean script writing.

* **Course Status:** Completed
* **Course Title:** Data Science Essentials with Python
* **Certificate / Proof of Completion:** 
  * Included in this repo under `./certificates/Data_Science_Essentials_Certificate.pdf` *(or replace with your credential link / screenshot)*.

---

## Task 02: NumPy Arrays and Numerical Computation

### B. Inputs and Mathematical Operations
* **Inputs:** A 1D array named `temperatures_c` storing 24 raw hourly ambient temperature readings recorded in Celsius.
* **Math operations:**
  * Arithmetic average via `np.mean()`, which sums all 24 elements and divides by 24: $\mu = \frac{1}{N}\sum x_i$.
  * Minimum and maximum values using `np.min()` and `np.max()` to find the diurnal extremes.
  * Vectorized linear conversion using $F = 1.8 \times C + 32$. Instead of writing a slow `for` loop to convert each reading, NumPy broadcasts the scalar math across the entire array at the C level.

### C. Parameter Variation Experiment
* **What I tweaked:** I went into the original script and changed the coldest recorded reading (index 3) from `17.5°C` down to `10.0°C` (a sudden 7.5-degree drop).
* **What happened to the output:**
  * The minimum dropped straight from `17.50°C` to `10.00°C`.
  * The daily mean dropped from `23.40°C` to `23.09°C`.
* **Why it changed:** The mean distributes weight evenly across all 24 data points. A reduction of $7.5$ on one point reduces the global average by $\frac{7.5}{24} \approx 0.3125^\circ\text{C}$.

### D. Output Interpretation
* The script gave an average temperature of **23.40°C**, showing typical warm conditions across the 24-hour cycle.
* The temperatures hit a low of **17.50°C** (nighttime/early morning) and climbed to **29.40°C** around early afternoon.
* The converted Fahrenheit values printed as `[65.3, 64.4, 64.04, 63.5, 64.22, 66.2]`, confirming the element-wise formula converted the whole array without throwing index or dimension errors.

### E. Real-World Adaptation: Network Bandwidth Throughput
* **Script Location:** `adaptations/Task_02_adapted.py`
* **Real-world Problem:** Instead of weather data, I adapted the logic to monitor 24-hour network traffic on an enterprise router to find bandwidth bottlenecks and calculate total daily data transfer.
* **Assumptions:**
  * The router samples average throughput in Mbps (Megabits per second) every hour.
  * 1 Byte = 8 bits, and 1 GB = 1,000 MB (decimal storage standard).
  * Conversion factor to get Gigabytes transferred in 1 hour: 
    $$\text{GB/hr} = \text{Mbps} \times \frac{3600 \text{ seconds}}{8000 \text{ Mbits/GB}} = \text{Mbps} \times 0.45$$
* **New Inputs:** 24 hourly readings ranging from 25.0 Mbps during off-peak night hours to 312.0 Mbps during peak daytime usage.
* **Results & Findings:**
  * **Daily Mean Throughput:** 174.58 Mbps
  * **Off-peak / Peak:** Min was 25.00 Mbps, peak hit 312.00 Mbps.
  * **Total Transferred Data:** `np.sum(hourly_gigabytes)` yielded **1,885.44 GB** transferred over the day.
* **Takeaway:** This gives the network admin clear visibility into peak congestion windows and helps verify whether our daily payload is approaching data cap limits.

## Task 03: Mathematical Functions and Vectorization

### B. Inputs and Mathematical Operations
* **Inputs:** A simulated dataset of 1,000,000 floating-point numbers (`latency_ms`) sampled uniformly between 5 ms and 250 ms using `np.random.default_rng(42)`.
* **Math operations:**
  * Scaling: Converting millisecond readings to seconds ($s = \frac{ms}{1000}$).
  * Non-linear mapping: Squaring each scaled reading ($s^2$).
  * Vectorization vs Iteration: Standard Python iterates through 1M items one by one with `.append()`, suffering dynamic type-checking overhead. NumPy pushes the entire batch straight to compiled C code (SIMD architecture), calculating all 1M operations in parallel.

### C. Parameter Variation Experiment
* **What I changed:** Scaled the sample count up from 1,000,000 to 3,000,000 measurements.
* **What happened:**
  * The manual loop execution time ballooned from ~0.45s to well over 1.4s.
  * The vectorized NumPy execution barely moved, staying around ~0.02s.
* **Why it changed:** As $N$ grows, the interpreter overhead in Python loops compounds linearly with every iteration, whereas NumPy executes contiguous memory operations with optimized memory caching.

### D. Output Interpretation
* The script outputs the first 5 squared latency values to confirm correct calculation.
* Comparing `Loop time` against `Vectorized time` showed an immediate ~20x-30x speed-up, proving why raw loops should never be used on heavy scientific or telemetry datasets.

### E. Real-World Adaptation: Network Packet Size Normalization
* **Script Location:** `adaptations/Task_03_adapted.py`
* **Real-world Problem:** Processing 1,000,000 network packet sizes captured by an intrusion detection system (IDS) to normalize them between $[0.0, 1.0]$ for an ML anomaly detection model.
* **Assumptions:** Standard Ethernet MTU frames range from 64 Bytes minimum to 1518 Bytes maximum.
* **New Inputs & Computations:** 
  * Random integer packet sizes generated between 64 and 1518.
  * Formula: $x_{\text{norm}} = \frac{x - 64}{1518 - 64}$.
* **Results & Findings:**
  * Iterative loop processing took ~0.38s.
  * Vectorized processing finished in ~0.008s (giving a ~47x performance gain).
* **Takeaway:** For real-time threat detection or streaming pipelines handling millions of packets per second, vectorized operations are essential to avoid dropping packets.

## Task 04: Floating-Point Arithmetic and Numerical Errors

### B. Inputs and Mathematical Operations
* **Inputs:** Decimals `0.1`, `0.2`, `0.3`, approximate square root value `1.414` against true $\sqrt{2}$, and pertubations $x \in \{10^{-4}, 10^{-8}, 10^{-12}\}$.
* **Math operations:**
  * Absolute Error: $E_{\text{abs}} = \vert{}x_{\text{true}} - x_{\text{approx}}\vert{}$.
  * Relative Error: $E_{\text{rel}} = \frac{\vert{}x_{\text{true}} - x_{\text{approx}}\vert{}}{\vert{}x_{\text{true}}\vert{}}$.
  * IEEE 754 Floating-point comparison: Using `np.isclose()` with machine epsilon tolerances instead of strict identity (`==`).
  * Catastrophic cancellation avoidance: Evaluating $\frac{\sqrt{1+x}-1}{x}$ algebraically restructured by conjugate multiplication to $\frac{1}{\sqrt{1+x}+1}$.

### C. Parameter Variation Experiment
* **What I tested:** Added an even smaller perturbation $x = 10^{-16}$ to the stability loop.
* **What happened:**
  * The direct formulation output crashed completely to `0.000000000000`.
  * The algebraically stable formulation retained the mathematically correct limit of `0.500000000000`.
* **Why it changed:** In 64-bit binary floating-point representation, $1.0 + 10^{-16}$ falls below the machine precision limit (around $2.22 \times 10^{-16}$), so it rounds straight back to $1.0$. Evaluating $(1.0 - 1.0) / 10^{-16}$ causes severe catastrophic cancellation, turning a non-zero quantity into an absolute zero numerator.

### D. Output Interpretation
* `0.1 + 0.2 == 0.3` returned `False` because binary cannot represent decimal tenths precisely (it stores `0.30000000000000004`), proving why financial or scientific systems require `np.isclose()`.
* The absolute error between $\sqrt{2}$ and $1.414$ was ~$0.00021356$, with a relative error of ~$0.0151\%$.
* The stable formulation consistently preserved precision across all small values of $x$.

### E. Real-World Adaptation: Financial Ledger Drift Analysis
* **Script Location:** `adaptations/Task_04_adapted.py`
* **Real-world Problem:** Tracking micro-transaction fees in high-frequency trading where binary rounding drift can accumulate substantial financial discrepancy across millions of records.
* **Assumptions:** 1,000,000 transactions each incurring a 0.05% fee on a $10.10 base charge.
* **New Inputs & Computations:**
  * Theoretical baseline: $\$5,050.000000$.
  * Comparing naive incremental loop addition vs NumPy's parallel precision accumulation.
* **Results & Findings:**
  * Naive incremental loop drifted slightly from the expected sum due to accumulated float round-off errors.
  * Direct equality `naive_total == exact_total_fees` evaluated to `False`, while `np.isclose()` confirmed ledger consistency within tolerance.
* **Takeaway:** Scientific and fintech code must never use exact equality checks on floating-point totals, and mathematical formulas must be factored algebraically to prevent precision loss.
