import numpy as np
import math
from collections import Counter

def symbolize(time_series, num_bins=2):
    """
    Convert a time series to a symbolic sequence.
    
    Args:
    time_series (list or numpy.array): The input time series.
    num_bins (int): Number of bins to use for symbolization. Default is 2.
    
    Returns:
    list: Symbolic sequence.
    """
    min_val, max_val = min(time_series), max(time_series)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    return np.digitize(time_series, bin_edges[1:-1]).tolist()

# Lempel-Ziv Complexity functions
def lempel_ziv_complexity(symbolic_sequence):
    """
    Calculate the Lempel-Ziv complexity of a symbolic sequence.
    
    Args:
    symbolic_sequence (list): The input symbolic sequence.
    
    Returns:
    int: The Lempel-Ziv complexity.
    """
    n = len(symbolic_sequence)
    i, j = 0, 1
    vocabulary = set()
    c = 1

    while j < n:
        substring = ''.join(map(str, symbolic_sequence[i:j+1]))
        if substring in vocabulary:
            j += 1
        else:
            vocabulary.add(substring)
            i = j
            j += 1
            c += 1

    return c

def normalized_lz_complexity(symbolic_sequence):
    """
    Calculate the normalized Lempel-Ziv complexity.
    
    Args:
    symbolic_sequence (list): The input symbolic sequence.
    
    Returns:
    float: The normalized Lempel-Ziv complexity.
    """
    n = len(symbolic_sequence)
    alpha = len(set(symbolic_sequence))
    c = lempel_ziv_complexity(symbolic_sequence)
    return (c / n) * math.log(n, alpha)

def lz_complexity_time_series(time_series, num_bins=2):
    """
    Calculate the Lempel-Ziv complexity for a time series.
    
    Args:
    time_series (list or numpy.array): The input time series.
    num_bins (int): Number of bins to use for symbolization. Default is 2.
    
    Returns:
    tuple: (LZ complexity, Normalized LZ complexity)
    """
    symbolic_sequence = symbolize(time_series, num_bins)
    lz = lempel_ziv_complexity(symbolic_sequence)
    normalized_lz = normalized_lz_complexity(symbolic_sequence)
    return lz, normalized_lz

# Effort-To-Compress Complexity functions
def nsrps_iteration(sequence):
    """
    Perform one iteration of the Non-sequential Recursive Pair Substitution algorithm.
    
    Args:
    sequence (list): The input symbolic sequence.
    
    Returns:
    tuple: (Transformed sequence, Whether the sequence changed, Replaced pair)
    """
    pairs = [''.join(map(str, sequence[i:i+2])) for i in range(len(sequence)-1)]
    pair_counts = Counter(pairs)
    
    if not pair_counts:
        return sequence, False, None

    most_common_pair = max(pair_counts, key=pair_counts.get)
    new_symbol = max(sequence) + 1
    
    new_sequence = []
    i = 0
    changed = False
    while i < len(sequence):
        if i < len(sequence) - 1 and f"{sequence[i]}{sequence[i+1]}" == most_common_pair:
            new_sequence.append(new_symbol)
            i += 2
            changed = True
        else:
            new_sequence.append(sequence[i])
            i += 1
    
    return new_sequence, changed, most_common_pair

def etc_complexity(symbolic_sequence):
    """
    Calculate the Effort-To-Compress complexity of a symbolic sequence.
    
    Args:
    symbolic_sequence (list): The input symbolic sequence.
    
    Returns:
    int: The ETC complexity (number of iterations).
    """
    sequence = symbolic_sequence.copy()
    iterations = 0
    
    while len(sequence) > 1:
        sequence, changed, replaced_pair = nsrps_iteration(sequence)
        if not changed or len(set(sequence)) == 1:
            break
        iterations += 1
        
        # Debugging information
        # print(f"Iteration {iterations}:")
        # print(f"  Replaced pair: {replaced_pair}")
        # print(f"  New sequence: {sequence}")
        # print(f"  Sequence length: {len(sequence)}")
        # print(f"  Unique symbols: {len(set(sequence))}")
        # print()
    
    return iterations


def normalized_etc_complexity(symbolic_sequence):
    """
    Calculate the normalized Effort-To-Compress complexity.
    
    Args:
    symbolic_sequence (list): The input symbolic sequence.
    
    Returns:
    float: The normalized ETC complexity.
    """
    n = len(symbolic_sequence)
    etc = etc_complexity(symbolic_sequence)
    return etc / (n - 1)

def etc_complexity_time_series(time_series, num_bins=2):
    """
    Calculate the Effort-To-Compress complexity for a time series.
    
    Args:
    time_series (list or numpy.array): The input time series.
    num_bins (int): Number of bins to use for symbolization. Default is 2.
    
    Returns:
    tuple: (ETC complexity, Normalized ETC complexity)
    """
    symbolic_sequence = symbolize(time_series, num_bins)
    etc = etc_complexity(symbolic_sequence)
    normalized_etc = normalized_etc_complexity(symbolic_sequence)
    return etc, normalized_etc

# Utility function for generating logistic map
def logistic_map(r, x0, n):
    """
    Generate a logistic map time series.
    
    Args:
    r (float): The r parameter in the logistic map equation.
    x0 (float): The initial condition (between 0 and 1).
    n (int): The number of iterations.
    
    Returns:
    list: The logistic map time series.
    """
    x = x0
    time_series = [x]
    for _ in range(n - 1):
        x = r * x * (1 - x)
        time_series.append(x)
    return time_series

# Example usage
if __name__ == "__main__":
    # Generate a logistic map time series
    r = 3.9  # Chaos regime
    x0 = 0.1
    n = 1000
    logistic_series = logistic_map(r, x0, n)

    # Calculate Lempel-Ziv complexity
    lz, normalized_lz = lz_complexity_time_series(logistic_series, num_bins=2)
    print(f"Lempel-Ziv Complexity: {lz}")
    print(f"Normalized Lempel-Ziv Complexity: {normalized_lz}")

    # Calculate Effort-To-Compress complexity
    etc, normalized_etc = etc_complexity_time_series(logistic_series, num_bins=2)
    print(f"Effort-To-Compress Complexity: {etc}")
    print(f"Normalized Effort-To-Compress Complexity: {normalized_etc}")