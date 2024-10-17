import numpy as np
import math
from collections import Counter
from utilities import *

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

def run_tests():
    test_cases = [
        ("aaa", 2, "Simple repetition"),
        ("abc", 3, "No repetition"),
        ("abcabc", 4, "Repeating pattern"),
        ("aabcaabcaab", 5, "Overlapping patterns"),
        ("", 0, "Empty string"),
        ("a", 1, "Single character"),
        ("abababab", 3, "Alternating pattern"),
        ("aaaaabbbbb", 3, "Two blocks of repetition"),
        ("abcdefghijklmnopqrstuvwxyz", 26, "Alphabet, no repetition"),
        ("thequickbrownfoxjumpsoverthelazydog", 29, "Pangram"),
        ("11010001", 5, "Binary sequence"),
        ("あいうえおあいうえお", 6, "Non-ASCII characters"),
        ("mississippi", 6, "Word with repetitions"),
    ]

    for sequence, expected_complexity, description in test_cases:
        complexity = lempel_ziv_complexity(sequence)
        result = "PASS" if complexity == expected_complexity else "FAIL"
        print(f"{result}: {description}")
        print(f"  Sequence: {sequence}")
        print(f"  Expected: {expected_complexity}, Got: {complexity}\n")
        
# Example usage
if __name__ == "__main__":
    # Generate a logistic map time series
    # r = 3.9  # Chaos regime
    # x0 = 0.1
    # n = 1000
    # logistic_series = logistic_map(r, x0, n)

    # # Calculate Lempel-Ziv complexity
    # lz, normalized_lz = lz_complexity_time_series(logistic_series, num_bins=2)
    # print(f"Lempel-Ziv Complexity: {lz}")
    # print(f"Normalized Lempel-Ziv Complexity: {normalized_lz}")

    # # Calculate Effort-To-Compress complexity
    # etc, normalized_etc = etc_complexity_time_series(logistic_series, num_bins=2)
    # print(f"Effort-To-Compress Complexity: {etc}")
    # print(f"Normalized Effort-To-Compress Complexity: {normalized_etc}")
    
    # run_tests()
    
    print(lempel_ziv_complexity("pprqprqp"))        # 4
    # print(etc_complexity("11010010"))               # 5
    