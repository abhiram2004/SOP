

# Dynamical Complexity of Short and Noisy Time Series

This repository contains the implementation of the research paper: **"Dynamical Complexity of Short and Noisy Time Series: Compression-Complexity vs. Shannon Entropy"**. The notebook `Implementation.ipynb` replicates the results and plots presented in the paper, step by step, according to the order of tables and figures in the research paper.

## Contents

### 1. Introduction
The paper investigates various complexity measures (Shannon Entropy, Lempel-Ziv Complexity, and Effort-To-Compress) for analyzing the dynamical complexity of chaotic systems, especially short and noisy time series. The focus is on demonstrating the superior performance of compression-based measures such as Lempel-Ziv (LZ) and Effort-To-Compress (ETC) over Shannon Entropy (H).

### 2. Implemented Systems and Methods
The chaotic systems under investigation are:
- Logistic Map
- Hénon Map
- Lorenz System
- Two-state Markov Chains

The complexity measures calculated for these systems include:
- **Shannon Entropy (H):** Measures the randomness in the symbolic sequence generated from the chaotic time series.
- **Lempel-Ziv Complexity (LZ):** Evaluates the rate of generation of new patterns.
- **Effort-To-Compress (ETC):** A compression-based measure using Non-sequential Recursive Pair Substitution (NSRPS) algorithm.

### 3. Code Structure

Each section in the notebook follows the same order of figures and tables as in the research paper.

- **Table 1: Analysis of Chaotic Systems**
  - Reproduces the complexity measures for Logistic Map, Hénon Map, and Lorenz System with different parameter settings.
  
- **Table 2: Minimum Data Length Without Noise**
  - Evaluates the minimum data length required to correctly order sequences hierarchically by their complexity, for different chaotic systems.
  
- **Table 3: Minimum Data Length With Noise**
  - Assesses the effect of noise (measured by SNR) on the performance of each complexity measure (H, LZ, and ETC).
  
- **Table 4: Short Binary Sequence Analysis**
  - Calculates complexity measures for binary sequences of varying lengths (from 4 to 16). Displays distinct values and the mean complexity for each measure.

- **Figure 1 to Figure 7**
  - The notebook includes the code and results for all the figures from the paper, including:
    - **Figure 1:** Shannon Entropy and Lyapunov Exponent vs. bifurcation parameter.
    - **Figure 2:** Correlation of measures with Lyapunov Exponent.
    - **Figure 3 to Figure 4:** Comparison of LZ and ETC with Lyapunov Exponent.
    - **Figure 5 to Figure 7:** Analysis of complexity measures on Two-State Markov Chains, showcasing convergence rates and standard deviation over time.

### 4. Final Notebook: `Implementation.ipynb`

The notebook `Implementation.ipynb` includes the following:
- Detailed code implementations for all systems and complexity measures.
- Step-by-step instructions and calculations.
- Reproduction of all results from the paper.
- Visualizations for each figure and table in the research paper.

### 5. Usage

To execute the code and reproduce the results:

1. Clone this repository.
2. Ensure you have the necessary dependencies installed:
   ```bash
   pip install numpy matplotlib scipy
   ```
3. Open the `Implementation.ipynb` notebook in Jupyter and run the cells in sequence.

### 6. Conclusion
This repository successfully replicates all key results from the paper and provides a comprehensive framework for analyzing the complexity of short and noisy time series using modern compression-complexity measures.

### 7. Acknowledgements
This work is based on the paper: **"Dynamical Complexity of Short and Noisy Time Series: Compression-Complexity vs. Shannon Entropy"**. Special thanks to the original authors for their contributions to the field of complexity measures.
```
