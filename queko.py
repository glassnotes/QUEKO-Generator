"""Generate 'quantum mapping examples with known optimal' (QUEKO) circuits."""
import numpy as np
import networkx as nx


def _backbone_construction(node_list, edge_list, depth, max_2q_gates):
    """Step 1 of QUEKO construction: build the "backbone" of depth T by
    creating a dependency chain of gates.

    Args:
        node_list (list[int]): The list of nodes in the hardware graph.
        edge_list (list[(int, int)]): A list of edges in the graph.
        depth (int): The desired depth.
        max_2q_gates (int): The maximum number of two-qubit gates.

    Returns:
        list[(int, (int, int))], int, int: A list of depth elements representing
        a set of gates denoted by (timestep, (qubit_1, qubit_2)), where qubit_2
        may not be present if the gate is a single-qubit gate. The other two
        returned integers represent the number of single-qubit gates and 2-qubit
        gates present in the circuit.
    """
    # Running count of gates
    count_1q_gates = 0
    count_2q_gates = 0

    edge_choices = list(range(len(edge_list)))

    # Results
    timesteps = []  # Timesteps
    gates = []  # Applied gates

    for timestep in range(depth):
        # Determine whether to add a 1- or 2-qubit gate
        gate_type = np.random.choice([1, 2])

        # Choose a random edge for 2-qubit gate, only if we haven't reached the max.
        if gate_type == 2 and count_2q_gates < max_2q_gates:
            which_qubits = edge_list[np.random.choice(edge_choices)]
        else:
            # Otherwise, choose a random qubit for single-qubit gate
            which_qubits = (np.random.choice(node_list),)

        # To create a dependency chain, we need there to be overlap between the
        # current gate and the previous one. If there is no overlap, pick again
        if timestep > 0:
            while not any(q in gates[timestep - 1] for q in which_qubits):
                if gate_type == 2 and count_2q_gates < max_2q_gates:
                    which_qubits = edge_list[np.random.choice(edge_choices)]
                else:
                    which_qubits = (np.random.choice(node_list),)

        # Update the gate counts and the list
        if gate_type == 2 and count_2q_gates < max_2q_gates:
            count_2q_gates += 1
        else:
            count_1q_gates += 1

        timesteps.append(timestep)
        gates.append(which_qubits)

    # Combined timesteps and gates
    gate_list = list(zip(timesteps, gates))

    return gate_list, count_1q_gates, count_2q_gates


def _sprinkling_phase(
    gate_list,
    node_list,
    edge_list,
    depth,
    max_1q_gates,
    max_2q_gates,
    count_1q_gates,
    count_2q_gates,
):
    """Step 2 of QUEKO construction: sprinkle gates in the empty spaces
    created by the backbone, up to a certain density.

    Args:
        gate_list (list[(int, (int, int))]): The list of gates representing the
            current circuit.
        node_list (list[int]): The list of nodes in the hardware graph.
        edge_list (list[(int, int)]): A list of edges in the graph.
        depth (int): The desired depth.
        max_1q_gates (int): The maximum number of single-qubit gates.
        max_2q_gates (int): The maximum number of two-qubit gates.
        count_1q_gates (int): The current number of single-qubit gates.
        count_2q_gates (int): The current number of two-qubit gates.

    Returns:
        list[(int, (int, int))]: A list representing a set of gates denoted by
        (timestep, (qubit_1, qubit_2)), where qubit_2 may not be present if the
        gate is a single-qubit gate.
    """
    available_timesteps = list(range(depth))
    edge_choices = list(range(len(edge_list)))

    for _ in range(depth, max_1q_gates + max_2q_gates):
        gate_type = np.random.choice([1, 2])

        overlap = True

        # Keep choosing random locations until there is no overlap
        while overlap is True:
            if gate_type == 2 and count_2q_gates < max_2q_gates:
                timestep = np.random.choice(available_timesteps)
                which_qubits = edge_list[np.random.choice(edge_choices)]
            else:
                timestep = np.random.choice(available_timesteps)
                which_qubits = (np.random.choice(node_list),)

            gates_at_t = [gate for gate in gate_list if gate[0] == timestep]
            overlap = any(any(q in gate[1] for q in which_qubits) for gate in gates_at_t)

        # Update the gate counts and the list
        if gate_type == 2:
            count_2q_gates += 1
        else:
            count_1q_gates += 1

        # Once selected, add the gates
        gate_list.append((timestep, which_qubits))

    return gate_list


def _scrambling_phase(gate_list, perm):
    """Step 3 of QUEKO construction: permute the order of the qubits.

    Args:
        gate_list (list[(int, (int, int))]): The list of gates representing the
            current circuit.
        perm (list[int]): A permuted order of qubits.

    Returns:
        list[(int, (int, int))]: A list representing a set of gates
        denoted by (timestep, (qubit_1, qubit_2)), where qubit_2 may not be
        present if the gate is a single-qubit gate, and the permutation of qubits
        given by perm applied.
    """
    permuted_gate_list = []

    for timestep, gate in gate_list:
        # Two-qubit gate
        if len(gate) == 2:
            permuted_gate = (perm[gate[0]], perm[gate[1]])
        else:
            permuted_gate = (perm[gate[0]],)

        permuted_gate_list.append((timestep, permuted_gate))

    return permuted_gate_list


def _generate_qasm(gate_list, n_qubits, gate_options={1: "x", 2: "cx"}):
    """Convert a list of gates into a QASM string.

    Args:
        gate_list (list[(int, (int, int))]): The list of gates representing the
            current circuit.
        n_qubits (int): The number of nodes in the graph.
        gate_options (Dict[(int, str)]): Which single- and two-qubit gates to
            use in the output QASM file. Defaults to "x" for single-qubit gates,
            and "cx" for two-qubit.

    Returns:
        str: QASM output for the circuit.
    """

    qasm_str = "OPENQASM 2.0;\n"
    qasm_str += 'include "qelib1.inc";\n'
    qasm_str += f"qreg q[{n_qubits}];\n"

    for _, gate in gate_list:
        if len(gate) == 2:
            qasm_str += f"{gate_options[2]} q[{gate[0]}], q[{gate[1]}];\n"
        else:
            qasm_str += f"{gate_options[1]} q[{gate[0]}];\n"

    return qasm_str


def queko_circuit(graph, depth, density_vec, gate_options={1: "x", 2: "cx"}, lattice_dim=0):
    """Generate a QUEKO circuit for a given graph with a target depth and gate density.

    Args:
        graph (nx.Graph): A hardware graph.
        depth (int): Target depth of the circuit.
        density_vec (list[float]): A two-element array containing the desired
            densities of single- and two-qubit gates.
        gate_options (Dict[(int, str)]): Which single- and two-qubit gates to
            use in the output QASM file. Defaults to "x" for single-qubit gates,
            and "cx" for two-qubit.
        lattice_dim (int): Special parameter for when the nodes in the input
            graph are doubly indexed due to, e.g., being constructed using
            nx.grid_graph. Represents the dimension of the lattice, so
            lattice_dim=7 means a 7x7 lattice of qubits with nearest-neighbour
            connections.

    Returns:
        str, list[int]: The QASM string for the circuit, and a list of integers
        representing the optimal mapping.

    Raises:
        ValueError: If the input parameters would return an inadmissible circuit.
    """
    n_qubits = len(graph.nodes)  # Number of nodes in the graph

    nodes = list(graph.nodes)
    edges = list(graph.edges)

    # Convert the nodes/edges to a list of integer tuples; for some cases, e.g., grid graphs,
    # the nodes are actually indexed by tuples (i, j) instead of integers
    if lattice_dim != 0:
        nodes = []
        # Index nodes starting from the top left corner and increase to the right
        for node in list(graph.nodes):
            nodes.append(node[0] * lattice_dim + node[1])

        edges = []
        for edge in list(graph.edges):
            i = edge[0][0] * lattice_dim + edge[0][1]
            j = edge[1][0] * lattice_dim + edge[1][1]
            edges.append((i, j))

    # Determine the number of single- and two-qubit gates required
    max_1q_gates = int(np.ceil(density_vec[0] * n_qubits * depth))
    max_2q_gates = int(np.ceil(density_vec[1] * n_qubits * depth / 2))

    # Determine whether the provided depth and density produces an admissible
    # circuit. max_1q_gates + max_2q_gates cannot be less than circuit depth,
    # otherwise not enough gates to produce something with depth T
    if max_1q_gates + max_2q_gates < depth:
        raise ValueError(
            "Input data inadmissible. Insufficient gate densities "
            f"to produce a circuit with depth {depth}.\n"
            f"max_1q_gates = {max_1q_gates}, max_2q_gates = {max_2q_gates}"
        )

    # max_1q_gates + 2 * max_2q_gates cannot be greater than number of qubits *
    # depth; that would be too many gates to fit in that depth on the device
    if max_1q_gates + 2 * max_2q_gates > n_qubits * depth:
        raise ValueError(
            "Input data inadmissible. Desired gate densities are too large "
            f"to produce a circuit with depth {depth}.\n"
            f"max_1q_gates = {max_1q_gates}, max_2q_gates = {max_2q_gates}"
        )

    # Number of two-qubit gates cannot be larger than the depth * size of the
    # maximal matching (otherwise, there are not enough disjoint edges to fit
    # all the two-qubit gates in the desired number of time steps)
    max_match_size = len(nx.maximal_matching(graph))
    if max_2q_gates > max_match_size * depth:
        raise ValueError(
            "Input data inadmissible. Number of 2-qubit gates determined"
            f"from density vector is too large to fit within depth {depth}.\n"
            f"max_2q_gates = {max_2q_gates}, max. matching size {max_match_size}"
        )

    # Generate a permutation of the graph vertices; this is the solution to the
    # allocation problem.
    perm = np.random.permutation(range(n_qubits))

    # Three stages of QUEKO construction
    gate_list, count_1q_gates, count_2q_gates = _backbone_construction(
        nodes, edges, depth, max_2q_gates
    )
    gate_list = _sprinkling_phase(
        gate_list, nodes, edges, depth, max_1q_gates, max_2q_gates, count_1q_gates, count_2q_gates
    )
    gate_list = _scrambling_phase(gate_list, perm)

    # Sort the resulting circuit according to the timesteps
    gate_list.sort(key=lambda x: x[0])

    qasm_string = _generate_qasm(gate_list, n_qubits, gate_options)

    return qasm_string, perm
