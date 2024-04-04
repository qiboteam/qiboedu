import matplotlib.pyplot as plt
import numpy as np

from qibo.config import raise_error

from qiboedu.scripts.utils import generate_bitstring_combinations

def visualize_states(counter, counter2=None):
    """Plot state's frequencies."""
        
    fig, ax = plt.subplots(figsize=(5, 5 * 6/8))
    
    ax.set_title('State visualization')
    ax.set_xlabel('States')
    ax.set_ylabel('#')
    plt.xticks(rotation=90)
       
    for i, state in enumerate(counter):
        if i == 0 and counter2 is not None:
            ax.bar(state, counter[state], color='#C194D8', edgecolor="black", label="Noiseless")
        else:
            ax.bar(state, counter[state], color='#C194D8', edgecolor="black")
            
    if counter2 is not None:
        for i, state in enumerate(counter2):
            if i == 0:
                ax.bar(state, counter2[state], color='orange', alpha=1, edgecolor="black", hatch="\\\\", facecolor="none", label="Noisy")
            else:
                ax.bar(state, counter2[state], color='orange', alpha=1, edgecolor="black", hatch="\\\\", facecolor="none")  
        plt.legend()

def plot_probabilities_from_state(state):
    """Plot amplitudes for a given quantum `state`."""

    bitstring = generate_bitstring_combinations(int(np.log2(len(state))))
    
    fig, ax = plt.subplots(figsize=(10, 5 * 6/8))
    
    ax.set_title('State visualization')
    ax.set_xlabel('States')
    ax.set_ylabel('Probability')
    
    for i, amp in enumerate(state):
        ax.bar(bitstring[i], np.abs(amp)**2, color='#C194D8', edgecolor="black")
        
    plt.xticks(rotation=90)

def plot_input_register_amplitudes(state, average_amp_value=None, title=None, save_as=None):
    """Plot amplitudes for a given quantum `state`."""    

    bitstring = generate_bitstring_combinations(int(np.log2(len(state))))

    fig, ax = plt.subplots(figsize=(10, 5 * 6/8))

    img_title = "State visualization"

    if save_as is None:
        ax.set_title(img_title)
        ax.set_xlabel('States')
        ax.set_ylabel('Amplitudes')

    amplitudes = []

    for i in range(0, len(state), 2):
        amplitudes.append((1/np.sqrt(2))*(np.real(state[i]) - np.real(state[i+1])))

    for i, amp in enumerate(amplitudes):
        # value = np.abs(amp)
        # print(amp)
        ax.bar(bitstring[i][1:], amp, color='#C194D8', edgecolor="black")

    if average_amp_value is not None:
        plt.hlines(average_amp_value, bitstring[0][1:], bitstring[-1][1:], color="black", ls="--", lw=2, label="Average value")
        plt.legend(loc=4, fontsize=12)
        
    plt.xticks(rotation=90, fontsize=12)  
    plt.yticks(fontsize=12)  

    for spine in ax.spines.values():
        spine.set_linewidth(1.5) 

    
    if save_as is not None:
        plt.savefig(f"../figures/amplitudes/x_amps_{save_as}.png", transparent=True, bbox_inches="tight")
    

def plot_probabilities(c, nshots, grover=False):
    """Plot probabilities for a given circuit `c` and a number of shots `nshots`."""

    bitstring = generate_bitstring_combinations(c.nqubits)
    frequencies = c(nshots=nshots).frequencies(binary=True)
    
    fig, ax = plt.subplots(figsize=(10, 5 * 6/8))
    
    ax.set_title('Probabilities')
    ax.set_xlabel('States')
    ax.set_ylabel('Prob')
    
    for i, p in enumerate(frequencies):
        if grover:
            bit = bitstring[i][1:]
        else:
            bit = bitstring[i]
        ax.bar(bit, frequencies[p] / nshots, color='#C194D8', edgecolor="black")\
        
    plt.xticks(rotation=90)
    plt.show()

def plot_grover_probabilities(c, nshots):
    """Plot probabilities for a given circuit `c` and a number of shots `nshots`."""

    bitstring = generate_bitstring_combinations(c.nqubits)
    frequencies = c(nshots=nshots).frequencies(binary=True)
    
    fig, ax = plt.subplots(figsize=(10, 5 * 6/8))
    
    ax.set_title('Probabilities')
    ax.set_xlabel('States')
    ax.set_ylabel('Prob')
    
    for i, p in enumerate(frequencies):
        bit = bitstring[i][1:]
        ax.bar(bit, frequencies[p] / nshots, color='#C194D8', edgecolor="black")\
        
    plt.xticks(rotation=90)
    

def plot_amplitudes(state):
    """Plot amplitudes for a given quantum `state`."""

    bitstring = generate_bitstring_combinations(int(np.log2(len(state))))

    fig, ax = plt.subplots(figsize=(10, 5 * 6/8))

    ax.set_title('State visualization')
    ax.set_xlabel('States')
    ax.set_ylabel('Amplitudes')

    for i, amp in enumerate(state):
        ax.bar(bitstring[i], np.real(amp), color='#C194D8', edgecolor="black")

    plt.xticks(rotation=90)
    plt.show()


def plot_density_matrix(state):
    """
    Plot the density matrix of a circuit result. The argument ``state`` have to 
    be a ``qibo.state`` object obtained via density matrix simulation.
    """

    if (len(state.shape) == 1):
        raise_error(TypeError, "The given state is not obtained via density matrix simulation.")

    nqubits = int(np.log2(state.shape[0]))
    bitstrings = generate_bitstring_combinations(nqubits)


    plt.figure(figsize=(5, 5))
    plt.imshow(np.abs(state), cmap="PRGn", vmin=0, vmax=1)
    plt.xticks(ticks=np.arange(0,2**nqubits,1), labels=bitstrings)
    plt.yticks(ticks=np.arange(0,2**nqubits,1), labels=bitstrings)
    plt.colorbar()
    plt.show()  
