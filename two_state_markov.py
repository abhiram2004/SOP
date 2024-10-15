import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import math
from collections import Counter
import itertools
import string

from utilities import *

# ... [Previous functions remain unchanged] ...

def two_state_markov_chain(p01, p10, n):
    """Simulate a two-state Markov chain."""
    states = [0]
    for _ in range(1, n):
        if states[-1] == 0:
            states.append(1 if np.random.random() < p01 else 0)
        else:
            states.append(0 if np.random.random() < p10 else 1)
    return states

def entropy_rate_markov(p01, p10):
    """Calculate the entropy rate of a two-state Markov chain."""
    mu0 = p10 / (p01 + p10)
    mu1 = p01 / (p01 + p10)
    return - (mu0 * (p01 * np.log2(p01) + (1-p01) * np.log2(1-p01)) +
              mu1 * (p10 * np.log2(p10) + (1-p10) * np.log2(1-p10)))

def moving_window_analysis(data, measure_func, window_size=20):
    """Perform moving window analysis on the data."""
    n = len(data)
    results = []

    for i in range(0, n - window_size + 1, window_size):
        window = data[i:i+window_size]
        results.append(measure_func(window))
    return results

def compare_complexity_measures(p01, p10, max_length, num_iterations=50):
    """Compare LZ and ETC complexity measures for increasing data lengths."""
    lz_results = []
    etc_results = []
    
    for _ in range(num_iterations):
        chain = two_state_markov_chain(p01, p10, max_length)
        symbolic_chain = ''.join(['0' if s == 0 else '1' for s in chain])
        
        lz_values = moving_window_analysis(symbolic_chain, normalized_lz_complexity)
        etc_values = moving_window_analysis(symbolic_chain, normalized_etc_complexity)
        
        lz_results.append(lz_values)
        etc_results.append(etc_values)
    
    lz_mean = np.mean(lz_results, axis=0)
    etc_mean = np.mean(etc_results, axis=0)
    lz_std = np.std(lz_results, axis=0)
    etc_std = np.std(etc_results, axis=0)
    
    return lz_mean, etc_mean, lz_std, etc_std

def plot_complexity_comparison(lz_mean, etc_mean, lz_std, etc_std, entropy_rate):
    """Plot the comparison of LZ and ETC complexity measures."""
    blocks = np.arange(1, len(lz_mean) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(blocks, lz_mean, label='LZ', color='blue')
    plt.plot(blocks, etc_mean, label='ETC', color='red')
    plt.axhline(y=entropy_rate, color='green', linestyle='--', label='Entropy rate')
    
    plt.fill_between(blocks, lz_mean - lz_std, lz_mean + lz_std, alpha=0.2, color='blue')
    plt.fill_between(blocks, etc_mean - etc_std, etc_mean + etc_std, alpha=0.2, color='red')
    
    plt.xlabel('Block Number')
    plt.ylabel('Mean Normalized Complexity')
    plt.title('Convergence of ETC and LZ Complexity Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_standard_deviation(lz_std, etc_std):
    """Plot the standard deviation of LZ and ETC complexity measures."""
    blocks = np.arange(1, len(lz_std) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(blocks, lz_std, label='LZ', color='blue')
    plt.plot(blocks, etc_std, label='ETC', color='red')
    
    plt.xlabel('Block Number')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation of Complexity Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
p01, p10 = 0.1, 0.8
max_length = 15000
window_size = 20

entropy_rate = entropy_rate_markov(p01, p10)
lz_mean, etc_mean, lz_std, etc_std = compare_complexity_measures(p01, p10, max_length)

plot_complexity_comparison(lz_mean, etc_mean, lz_std, etc_std, entropy_rate)
plot_standard_deviation(lz_std, etc_std)