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