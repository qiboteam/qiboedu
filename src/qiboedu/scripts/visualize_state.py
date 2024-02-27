import matplotlib.pyplot as plt
import numpy as np

def visualize_states(counter):
    """
    Plot state's amplitudes.

    Args:
        counter (dict): frequencies counter object you get by executing a qibo circuit
            afer setting a certain ``nshots``.
    """
        
    fig, ax = plt.subplots(figsize=(5, 5 * 6/8))
    
    ax.set_title('State visualization')
    ax.set_xlabel('States')
    ax.set_ylabel('#')
    plt.xticks(rotation=90)
       
    for i, state in enumerate(counter):
            ax.bar(state, counter[state], color='purple', alpha=0.5, edgecolor="black")
            
    plt.show() 
