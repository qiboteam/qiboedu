import matplotlib.pyplot as plt

import numpy as np

from qibo.config import raise_error

from qiboedu.scripts.utils import generate_bitstring_combinations

def interpolate_cmap(palette_name, n):
    """Interpolating a colormap with N values."""
    cmap = plt.get_cmap(palette_name)
    return [cmap(i / (n - 1)) for i in range(n)]


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


def plot_vqe_states(state, state2=None):
    """
    Plot `state` and `state2` if provided. 
    # TODO: merge this function into one of the others.
    """
    n = int(np.log2(len(state)))
    bitstrings = []
    for i in range(2**n):
        bitstrings.append(format(i, f"0{n}b"))
    for i, amp in enumerate(state):
        if i == 0:
            plt.bar(bitstrings[i], np.abs(amp)**2, color='#C194D8', alpha=0.7, edgecolor="black", label="Ground state")
        else:
            plt.bar(bitstrings[i], np.abs(amp)**2, color='#C194D8', alpha=0.7, edgecolor="black")
    if state2 is not None:     
        for i, amp in enumerate(state2):
            if i == 0:
                plt.bar(bitstrings[i], np.abs(amp)**2, color='black', alpha=1, edgecolor="black", hatch="\\\\", facecolor="none", label="VQE approximation")
            else:
                plt.bar(bitstrings[i], np.abs(amp)**2, color='black', alpha=1, edgecolor="black", hatch="\\\\", facecolor="none")
                
    plt.xticks(rotation=90)
    plt.xlabel("Components")
    plt.ylabel("Probabilities")
    plt.title("State representation")
    plt.legend()
    plt.show()


def plot_bell_inequalities(experiment, Q_values, ac_steps, param_steps, param_label, y_bounds, plot_projection=None, img_width=1, legendloc=1, savetitle=None):
    """
    Plot the Bell inequalities' Q values and their corresponding classical bound. See details in the Bell-related notebooks.
    """
    colors = [
        "#CC79A7",
        "#D55E00",
        "#0072B2",
        "#F0E442",
        "#009E73",
        "#56B4E9",
        "#E69F00",
    ]

    if experiment == "bell64":
        if plot_projection is None:
            yfill_bounds = (2, 1)
        elif plot_projection == "polar":
            yfill_bounds = (1, 1.6)
        classical_bound = 1
        ylabel = r"$Q^B$"
    elif experiment == "bell-wigner":
        yfill_bounds = (0.2, 0)
        classical_bound = 0
        ylabel = r"$Q^W$" 
    elif experiment == "chsh":
        colors.extend(["#67E1F0", "#67F09D"])
        yfill_bounds = (2, 3)
        classical_bound = 2
        ylabel = r"$Q^S$"

    _, ax = plt.subplots(figsize=(10 * img_width, 10 * img_width * 5 / 8), subplot_kw={"projection": plot_projection})
    labels=[f'${param_label}/\pi = 0$'] + [f'${param_label}/\pi = {i_param}/{param_steps}$' for i_param in range(1, param_steps)] + [f'${param_label}/\pi = 1$']

    if plot_projection is None:
        th = [(i / ac_steps) for i in range(ac_steps + 1)]
    elif plot_projection == "polar":
        th = [(i * np.pi/ac_steps) for i in range(ac_steps+1)]
    else:
        raise ValueError(f"plot_projection {plot_projection} is not supported here, please use None or polar.")
    
    for i_param in range(param_steps+1):
        ax.plot(th, Q_values[i_param][:len(th)], color=colors[i_param], label=labels[i_param], lw=2, alpha=0.8)
    
    # fill between constraints
    ax.fill_between(x=th, y1=yfill_bounds[0], y2=yfill_bounds[1], color="0.5", alpha=0.3)

    if plot_projection is None:
        ax.hlines(classical_bound, min(th), max(th), color="black", ls="--", label="Classic bound", lw=1.5)
        plt.xlabel('$\\theta_{ac} / \pi$', fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.ylim(y_bounds[0], y_bounds[1])
        plt.grid(True)
        plt.legend(loc=legendloc, fontsize=16, ncols=3)
    
    elif plot_projection == "polar":
        ax.set_xticks([(i*np.pi/6) for i in range(7)], labels=['0'] + ['$%d\\pi/6$    ' % i for i in range(1, 6)] + ['$\\pi$'])
        ax.set_theta_zero_location("N")
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_ylim(y_bounds[0], y_bounds[1])
        ax.yaxis.set_tick_params(labelsize=16)
        ax.legend(bbox_to_anchor=(0.9, 1.0), fontsize=14)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if savetitle is not None:
        plt.savefig(f"{savetitle}.pdf", bbox_inches="tight")

    plt.show()
