from collections import Counter
import itertools
import string

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
import itertools
import string

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
        print(f"Step {step}: Sequence transformed to {''.join(sequence)}")
    
    return step

# Example usage:
input_sequence = "AABABBAB"
complexity = etc_complexity(input_sequence)
print(f"ETC Complexity: {complexity}")
