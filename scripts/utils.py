import numpy as np
from qibo import Circuit, gates

def generate_bitstring_combinations(n):
    """Generate all bitstring combinations given bitstring length `n`."""
    bitstrings = []
    for i in range(2**n):
        bitstrings.append(format(i, f"0{n}b"))
    return bitstrings

def compute_max_probability(c, nshots):
    """Compute max amplitude probability executing `c` with `nshots`."""
    outcome = c(nshots=nshots)
    n = c.nqubits
    probs = outcome.probabilities(qubits=range(n-1))
    return max(probs) 

def black_blox_oracle(n):
    """Black box oracle function, to be used in Lecture 3."""
    c = Circuit(n+1)
    # this is balanced!
    for q in range(n):
        c.add(gates.CNOT(q0=q, q1=n))
    return c

def compute_input_register_average_amplitude(state):
    """Compute the average amplitude considering only the input register."""
    amplitudes = []
    for i in range(0, len(state), 2):
        amplitudes.append((1/np.sqrt(2))*(np.real(state[i]) - np.real(state[i+1])))
    return np.mean(amplitudes)