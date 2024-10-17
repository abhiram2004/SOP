import numpy as np
from itertools import product
from utilities import *

def analyze_short_binary_sequences(min_length=4, max_length=16):
    """
    Analyze complexity measures for all possible binary sequences of lengths from min_length to max_length.
    
    Returns:
    - A dictionary with sequence lengths as keys and analysis results as values.
    """
    results = {}
    
    for length in range(min_length, max_length + 1):
        # Generate all possible binary sequences of the given length
        sequences = [''.join(map(str, seq)) for seq in product([0, 1], repeat=length)]
        
        # Initialize lists to store the results
        entropy_values = []
        lz_values = []
        etc_values = []
        
        for seq in sequences:
            # print(seq)
            # Compute Shannon entropy
            entropy_values.append(shannon_entropy(seq))
            
            # Compute normalized LZ complexity
            lz_values.append(normalized_lz_complexity(seq))
            
            # Compute normalized ETC complexity
            etc_values.append(normalized_etc_complexity(seq))
        
        # Compute the number of distinct levels for each measure
        entropy_levels = len(set(entropy_values))
        lz_levels = len(set(lz_values))
        etc_levels = len(set(etc_values))
        
        # Compute mean values
        entropy_mean = np.mean(entropy_values)
        lz_mean = np.mean(lz_values)
        etc_mean = np.mean(etc_values)
        
        # Store the results
        results[length] = {
            'entropy': {'levels': entropy_levels, 'mean': entropy_mean},
            'lz': {'levels': lz_levels, 'mean': lz_mean},
            'etc': {'levels': etc_levels, 'mean': etc_mean}
        }
    
    return results

# Run the analysis
analysis_results = analyze_short_binary_sequences()

# Print the results in a format similar to Table 4
print("Length | Entropy Levels | LZ Levels | ETC Levels | Entropy Mean | LZ Mean | ETC Mean")
print("-------|----------------|-----------|------------|--------------|---------|----------")
for length, data in analysis_results.items():
    print(f"{length:6d} | {data['entropy']['levels']:15d} | {data['lz']['levels']:10d} | {data['etc']['levels']:11d} | {data['entropy']['mean']:13.4f} | {data['lz']['mean']:7.4f} | {data['etc']['mean']:9.4f}")