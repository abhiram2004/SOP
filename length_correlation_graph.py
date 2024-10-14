from utilities import *



def compute_correlations(a_range, length, n_discard = 0):
    """Compute correlations for a given time series length."""
    lyapunov = []
    shannon = []
    lz = []
    etc = []

    for a in a_range:
        x0 = np.random.random()
        time_series = logistic_map(a, x0, length + n_discard)[n_discard:]
        
        lyap = lyapunov_exponent(a, time_series)
        lyapunov.append(lyap)
        
        symbolic_sequence = symbolize(time_series)
        
        shan = shannon_entropy(symbolic_sequence)
        shannon.append(shan)
        
        lz_comp = lempel_ziv_complexity(symbolic_sequence) / length
        lz.append(lz_comp)
        
        etc_comp = etc_complexity(symbolic_sequence) / (length - 1)
        etc.append(etc_comp)

    corr_shannon = np.corrcoef(lyapunov, shannon)[0, 1]
    corr_lz = np.corrcoef(lyapunov, lz)[0, 1]
    corr_etc = np.corrcoef(lyapunov, etc)[0, 1]

    return corr_shannon, corr_lz, corr_etc

def get_correlations(a_range, lengths, n_discard = 0):
    correlations = {
        'Shannon': [],
        'LZ': [],
        'ETC': []
    }

    for length in lengths:
        corr_shannon, corr_lz, corr_etc = compute_correlations(a_range, length, n_discard)
        correlations['Shannon'].append(corr_shannon)
        correlations['LZ'].append(corr_lz)
        correlations['ETC'].append(corr_etc)

    return correlations

def length_correlation_graph(correlations):
    
    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']
    for i, (measure, corrs) in enumerate(correlations.items()):
        plt.plot(lengths, corrs, marker=markers[i], label=measure)

    plt.xscale('log')
    plt.xlabel('Time Series Length')
    plt.ylabel('Correlation with Lyapunov Exponent')
    plt.title('Correlation with Lyapunov Exponent vs Time Series Length')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the correlation values
    for length, shannon, lz, etc in zip(lengths, correlations['Shannon'], correlations['LZ'], correlations['ETC']):
        print(f"Length: {length}")
        print(f"  Shannon: {shannon:.4f}")
        print(f"  LZ: {lz:.4f}")
        print(f"  ETC: {etc:.4f}")
        print()

if __name__ == "__main__":
    # Plot the graph between length of sequence and 
    a_range = np.linspace(3.55, 4.0, 50)
    n_discard = 0
    lengths = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    correlations = get_correlations(a_range, lengths, n_discard)
    length_correlation_graph(correlations)