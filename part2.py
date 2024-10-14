import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.integrate import odeint
import math
from collections import Counter
from utilities import *


def logistic_map(a, x0, n):
    """Generate a time series using the logistic map."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = a * x[i-1] * (1 - x[i-1])
    return x

def henon_map(a, b, x0, y0, n):
    """Generate a time series using the Hénon map."""
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in range(1, n):
        x[i] = 1 - a * x[i-1]**2 + y[i-1]
        y[i] = b * x[i-1]
    return x, y

def lorenz_system(state, t, sigma, rho, beta):
    """Lorenz system differential equations."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z    
    return [dx, dy, dz]

def generate_lorenz(sigma, rho, beta, x0, y0, z0, n, dt):
    """Generate a time series using the Lorenz system."""
    t = np.linspace(0, n*dt, n)
    initial_state = [x0, y0, z0]
    solution = odeint(lorenz_system, initial_state, t, args=(sigma, rho, beta))
    return solution

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
    sub_strings = set()
    i, n = 0, len(sequence)
    
    while i < n:
        j = i + 1
        while j <= n:
            sub = ''.join(sequence[i:j])
            if sub not in sub_strings:
                sub_strings.add(sub)
                i = j
                break
            j += 1
        if j == n + 1:
            break
    
    return len(sub_strings)

def normalized_lz_complexity(sequence):
    n = len(sequence)
    alpha = len(set(sequence))
    c = lempel_ziv_complexity(sequence)

    if alpha <= 1:
        return 0.0
    return (c / n) * math.log(n, alpha)

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

def lyapunov_logistic(a, x, n_discard=100):
    """Calculate the Lyapunov exponent for the logistic map."""
    return np.mean(np.log(np.abs(a * (1 - 2 * x[n_discard:]))))

import numpy as np

def lyapunov_henon(a, b, x, y, n_discard=0):
    """Calculate the Lyapunov exponent for the Hénon map."""
    n = len(x) - n_discard
    lyap = np.zeros(n)
    
    for i in range(n):
        # Calculate Jacobian for each point
        J = np.array([[-2*a*x[i+n_discard], 1],
                      [b, 0]])
        
        # Calculate eigenvalues
        eigs = np.linalg.eigvals(J)  # Direct eigenvalue calculation
        max_eig = np.max(np.abs(eigs))
        
        if max_eig > 0:
            lyap[i] = np.log(max_eig)
        else:
            lyap[i] = -np.inf  # Handle non-positive eigenvalues

    # Remove any -inf or NaN from lyap
    lyap = np.nan_to_num(lyap, nan=0.0, posinf=0.0)

    # Calculate and return the Lyapunov exponent
    return 0.5 * np.mean(lyap[lyap > -np.inf])  # Ensure we only take the mean of valid entries


def lyapunov_lorenz(sigma, rho, beta, trajectory, dt, n_discard=0):
    """Calculate the Lyapunov exponent for the Lorenz system."""
    
    def jacobian(X):
        x, y, z = X
        return np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])
    
    n = len(trajectory)
    Q = np.eye(3)  # Initialize Q as identity
    running_sum = 0.0

    for i in range(n_discard, n):
        J = jacobian(trajectory[i])
        Q, R = np.linalg.qr(J @ Q)
        
        # Check the diagonal elements of R before taking the logarithm
        diag_R = np.diag(R)
        
        # Replace any non-positive elements with a very small positive value
        diag_R = np.where(diag_R > 0, diag_R, np.nan)  # Replace non-positive values with NaN

        running_sum += np.nansum(np.log(np.abs(diag_R)))  # Sum logs of the diagonal elements of R while ignoring NaNs

    # Handle the case when n - n_discard is zero
    if (n - n_discard) * dt == 0:
        return np.nan  # Or return 0 or another value as appropriate

    return running_sum / ((n - n_discard) * dt)  # Return the average

# Parameters
n_points = 1000
n_trials = 30

# Logistic map parameters
logistic_a_values = [3.83, 3.9, 4.0]

# Hénon map parameters
henon_a_values = [1.2, 1.3, 1.4]
henon_b = 0.3

# Lorenz system parameters
lorenz_rho_values = [20, 25, 28]
lorenz_sigma = 10
lorenz_beta = 8/3
lorenz_dt = 0.01

# Initialize arrays to store results
systems = ['Logistic', 'Henon', 'Lorenz']
measures = ['LZ', 'ETC', 'Shannon', 'Lyapunov']
results = {sys: {measure: np.zeros((3, n_trials)) for measure in measures} for sys in systems}

def analyze_system(time_series, system_name, param_value, trial, **kwargs):
    symbolic_sequence = symbolize(time_series)
    
    results[system_name]['LZ'][param_value, trial] = normalized_lz_complexity(symbolic_sequence)
    results[system_name]['ETC'][param_value, trial] = normalized_etc_complexity(symbolic_sequence)
    results[system_name]['Shannon'][param_value, trial] = shannon_entropy(symbolic_sequence)
    
    # Calculate Lyapunov exponent based on the system
    if system_name == 'Logistic':
        results[system_name]['Lyapunov'][param_value, trial] = lyapunov_logistic(kwargs['a'], time_series)
    elif system_name == 'Henon':
        results[system_name]['Lyapunov'][param_value, trial] = lyapunov_henon(kwargs['a'], kwargs['b'], kwargs['x'], kwargs['y'])
    elif system_name == 'Lorenz':
        results[system_name]['Lyapunov'][param_value, trial] = lyapunov_lorenz(kwargs['sigma'], kwargs['rho'], kwargs['beta'], kwargs['trajectory'], kwargs['dt'])

# Generate data and calculate complexities
for trial in range(n_trials):
    # Logistic Map
    for i, a in enumerate(logistic_a_values):
        x0 = np.random.random()
        time_series = logistic_map(a, x0, n_points)
        analyze_system(time_series, 'Logistic', i, trial, a=a)
    
    # Hénon Map
    for i, a in enumerate(henon_a_values):
        x0, y0 = np.random.random(2)
        x, y = henon_map(a, henon_b, x0, y0, n_points)
        analyze_system(x, 'Henon', i, trial, a=a, b=henon_b, x=x, y=y)
    
    # Lorenz System
    for i, rho in enumerate(lorenz_rho_values):
        x0, y0, z0 = np.random.random(3)
        trajectory = generate_lorenz(lorenz_sigma, rho, lorenz_beta, x0, y0, z0, n_points, lorenz_dt)
        analyze_system(trajectory[:, 0], 'Lorenz', i, trial, sigma=lorenz_sigma, rho=rho, beta=lorenz_beta, trajectory=trajectory, dt=lorenz_dt)

# Plotting
plt.figure(figsize=(20, 15))
plot_index = 1

for system in systems:
    for measure in measures:
        plt.subplot(3, 4, plot_index)
        plt.boxplot([results[system][measure][i, :] for i in range(3)])
        plt.title(f'{system} - {measure}')
        if system == 'Logistic':
            plt.xticks(range(1, 4), [f'a = {a}' for a in logistic_a_values])
        elif system == 'Henon':
            plt.xticks(range(1, 4), [f'a = {a}' for a in henon_a_values])
        else:  # Lorenz
            plt.xticks(range(1, 4), [f'ρ = {rho}' for rho in lorenz_rho_values])
        plt.ylabel('Complexity')
        plot_index += 1

plt.tight_layout()
plt.savefig('chaotic_systems_complexity_analysis.png')
plt.close()

# Print average results
for system in systems:
    print(f"\n{system} System Results:")
    for i, param in enumerate(logistic_a_values if system == 'Logistic' else (henon_a_values if system == 'Henon' else lorenz_rho_values)):
        print(f"\nParameter = {param}:")
        for measure in measures:
            mean = np.mean(results[system][measure][i, :])
            std = np.std(results[system][measure][i, :])
            print(f"{measure}: {mean:.4f} ± {std:.4f}")