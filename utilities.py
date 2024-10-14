import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.integrate import odeint
import math

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

def normalized_lz_complexity(sequnce):
    n = len(symbolic_sequence)
    alpha = len(set(symbolic_sequence))
    c = lempel_ziv_complexity(symbolic_sequence)
    return (c / n) * math.log(n, alpha)

def etc_complexity(sequence):
    """Calculate the Effort-To-Compress complexity of a symbolic sequence."""
    iterations = 0
    while len(set(sequence)) > 1:
        pairs = [''.join(sequence[i:i+2]) for i in range(len(sequence)-1)]
        most_common = max(set(pairs), key=pairs.count)
        new_symbol = chr(max(ord(max(sequence)) + 1, 128))
        sequence = ''.join(sequence).replace(most_common, new_symbol)
        iterations += 1
    return iterations

def normalized_etc_complexity(sequence):
    n = len(symbolic_sequence)
    etc = etc_complexity(symbolic_sequence)
    return etc / (n - 1)

def shannon_entropy(sequence):
    """Calculate the Shannon entropy of a symbolic sequence."""
    _, counts = np.unique(sequence, return_counts=True)
    return entropy(counts, base=2)


# ... (previous functions remain the same) ...

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
    return solution[:, 0], solution[:, 1], solution[:, 2]

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

# Analysis function
def analyze_system(time_series, system_name, param_value, trial):
    symbolic_sequence = symbolize(time_series)
    
    results[system_name]['LZ'][param_value, trial] = normalized_lz_complexity(symbolic_sequence)
    results[system_name]['ETC'][param_value, trial] = normalized_etc_complexity(symbolic_sequence)
    results[system_name]['Shannon'][param_value, trial] = shannon_entropy(symbolic_sequence)
    
    # Note: Lyapunov exponent calculation is system-specific and not implemented here

# Generate data and calculate complexities
for trial in range(n_trials):
    # Logistic Map
    for i, a in enumerate(logistic_a_values):
        x0 = np.random.random()
        time_series = logistic_map(a, x0, n_points)
        analyze_system(time_series, 'Logistic', i, trial)
    
    # Hénon Map
    for i, a in enumerate(henon_a_values):
        x0, y0 = np.random.random(2)
        x, y = henon_map(a, henon_b, x0, y0, n_points)
        analyze_system(x, 'Henon', i, trial)  # Analyzing x-component
    
    # Lorenz System
    for i, rho in enumerate(lorenz_rho_values):
        x0, y0, z0 = np.random.random(3)
        x, y, z = generate_lorenz(lorenz_sigma, rho, lorenz_beta, x0, y0, z0, n_points, lorenz_dt)
        analyze_system(x, 'Lorenz', i, trial)  # Analyzing x-component

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
    print(f"\n{system} Map Results:")
    for i, param in enumerate(logistic_a_values if system == 'Logistic' else (henon_a_values if system == 'Henon' else lorenz_rho_values)):
        print(f"\nParameter = {param}:")
        for measure in measures:
            mean = np.mean(results[system][measure][i, :])
            std = np.std(results[system][measure][i, :])
            print(f"{measure}: {mean:.4f} ± {std:.4f}")