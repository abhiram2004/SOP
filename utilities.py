import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.integrate import odeint
import math
from collections import Counter
import itertools
import string

def logistic_map(a, x0, n):
    """Generate a time series using the logistic map."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = a * x[i-1] * (1 - x[i-1])
    return x

def lyapunov_exponent(a, x, n_discard=0):
    """Calculate the Lyapunov exponent for the logistic map."""
    return np.mean(np.log(np.abs(a * (1 - 2 * x[n_discard:]))))

def symbolize(time_series, num_bins=4):
    """Convert a time series to a symbolic sequence."""
    min_val, max_val = np.min(time_series), np.max(time_series)
    mean_val = np.mean(time_series)
    v1 = (min_val + mean_val) / 2
    v2 = (mean_val + max_val) / 2
    
    bins = [min_val, v1, mean_val, v2, max_val]
    symbols = ['A', 'B', 'C', 'D']
    
    return [symbols[np.digitize(x, bins[1:-1]) - 1] for x in time_series]

def lempel_ziv_complexity(sequence):
    """Calculate the Lempel-Ziv complexity of a symbolic sequence."""
    i, n = 0, len(sequence)
    patterns = []
    
    
    while i < n:
        for j in range(i + 1, n + 1):
            substr = sequence[i:j]
            if substr not in sequence[0:i]:
                patterns.append(substr)
                i = j
                break
        if j == n:
            break
    
    print("Sequence: ", sequence)
    print(patterns)
    return len(patterns)

def normalized_lz_complexity(sequence):
    n = len(sequence)
    alpha = len(set(sequence))
    c = lempel_ziv_complexity(sequence)

    if alpha <= 1:
        return 0.0
    normal_c = (c / n) * math.log(n, alpha)
    
    print("n: ", n)
    print("Alpha: ", alpha)
    print("LZ Complexity: ", c)
    print("Normal LZ: ", normal_c)

    return normal_c

def most_frequent_pair(sequence):
    """Find the most frequent pair in the sequence."""
    pair_counts = Counter()
    
    # Count all pairs in the sequence.
    for i in range(len(sequence) - 1):
        pair = (sequence[i], sequence[i + 1])  # Use tuples to represent pairs.
        pair_counts[pair] += 1
    
    # Find the pair with the maximum count.
    if not pair_counts:
        return None
    return max(pair_counts, key=pair_counts.get)

def replace_pair(sequence, pair, new_symbol):
    """Replace all occurrences of a given pair in the sequence with a new symbol."""
    i = 0
    new_sequence = []
    while i < len(sequence):
        # Check if the current and next symbols form the target pair.
        if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
            new_sequence.append(new_symbol)
            i += 2  # Skip the pair.
        else:
            new_sequence.append(sequence[i])
            i += 1
    return new_sequence

def next_symbol_generator():
    """Generate an infinite sequence of symbols."""
    # Use uppercase letters, then lowercase, and combine them as needed.
    characters = string.ascii_letters  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    
    # Generate single-letter symbols first.
    for char in characters:
        yield char
    
    # Generate combinations of increasing length.
    length = 2
    while True:
        for combination in itertools.product(characters, repeat=length):
            yield ''.join(combination)
        length += 1

# Create a generator instance.
symbol_gen = next_symbol_generator()

def next_symbol(counter=None):
    """Fetch the next available symbol from the generator."""
    return next(symbol_gen)

def etc_complexity(sequence):
    """Calculate the Entropy-based Transformation Complexity (ETC)."""
    # Convert sequence into a list of characters for mutability.
    sequence = list(sequence)
    current_symbols = set(sequence)
    step = 0
    
    while len(set(sequence)) > 1 and len(sequence) > 1:
        pair = most_frequent_pair(sequence)
        if pair is None:
            break  # No pairs to replace, end the process.
        
        # Generate the next available symbol.
        new_symbol = next_symbol(current_symbols)
        current_symbols.add(new_symbol)
        
        # Replace the pair with the new symbol.
        sequence = replace_pair(sequence, pair, new_symbol)
        step += 1
        # print(f"Step {step}: Sequence transformed to {''.join(sequence)}")
    
    return step

def normalized_etc_complexity(sequence):
    n = len(sequence)
    etc = etc_complexity(sequence)
    return etc / (n - 1)


def shannon_entropy(sequence):
    """Calculate the Shannon entropy of a symbolic sequence."""
    _, counts = np.unique(sequence, return_counts=True)
    return entropy(counts, base=2)
