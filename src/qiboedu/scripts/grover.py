from qibo import Circuit, gates

def build_initial_layer(n):
    """Build the initial superposition in Grover circuit."""
    # n+1 qubits circuit
    c = Circuit(n+1)
    # state preparation into |0>|0>...|1>
    c.add(gates.X(n))
    # add superposition
    for q in range(n+1):
        c.add(gates.H(q))
    return c


# final measurements
def build_final_layer(n):
    """Build the final layer of Grover with measurement gates."""
    # n+1 qubits circuit
    c = Circuit(n+1)
    # measurements
    c.add(gates.M(*range(n)))
    return c


def build_oracle(guilty_state, n, depth=None):
    """Swap the guilty state."""
    # n+1 qubits circuit
    circuit = Circuit(n+1)
    zeros = []
    for i, bit in enumerate(guilty_state):
        if bit == "0":
            zeros.append(i)
    [circuit.add(gates.X(q)) for q in zeros]
    if depth == 1: 
        return circuit
    # triggered only if the control state is |111111>
    circuit.add(gates.X(q=n).controlled_by(*range(n)))
    if depth == 2:
        return circuit
    [circuit.add(gates.X(q)) for q in zeros]
    return circuit

def build_diffusion_operator(n, depth=None):
    """Build Grover diffusion operator."""
    # n+1 qubits circuit
    circuit = Circuit(n+1)
    circuit.add(gates.X(n))
    if depth == 1:
        return circuit
    for q in range(n):
        circuit.add(gates.H(q=q))
    if depth == 2:
        return circuit
    for q in range(n):
        circuit.add(gates.X(q=q))
    if depth == 3:
        return circuit
    circuit.add(gates.X(q=n).controlled_by(*range(n)))
    if depth == 4:
        return circuit
    for q in range(n):
        circuit.add(gates.X(q=q))
    if depth == 5:
        return circuit
    for q in range(n):
        circuit.add(gates.H(q=q))
    return circuit


def build_grover(guilty_state, n, nsteps):
    """Build grover circuit with `n` qubits and repeating grover iteration `steps` times."""
    # n+1 qubits circuit
    c = Circuit(n+1)
    # initial layer
    c += build_initial_layer(n)
    # repeat nstepts time oracle + diffusion operator
    for _ in range(nsteps):
        # build_oracle requires guilty_state and n
        c += build_oracle(guilty_state, n)
        # build_diffusion operator requires just n
        c += build_diffusion_operator(n)
    # final layer
    c += build_final_layer(n)
    return c