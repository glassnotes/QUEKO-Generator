# QUEKO circuit generator

Generates *quantum mapping examples with known optimal* which can be used for benchmarking qubit allocation (placement) techniques. The tool is written in Python and follows Algorithm 1 in the [original QUEKO paper](https://arxiv.org/abs/2002.09783) by Tan and Cong.

## Requirements

 - `numpy`
 - `networkx`

## Example

QUEKO circuits are hardware-graph dependent, so we first construct a graph.

```pycon
>>> G = nx.Graph()
>>> G.add_nodes_from(list(range(5)))
>>> G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)])
```

Next, we choose a target depth (10), and a gate density vector ([0.2, 0.3]) for single- and two-qubit gates respectively.

```pycon
>>> qasm_string, perm = queko_circuit(G, 10, [0.2, 0.3])
```

The output consists of the representation of the circuit in QASM format:

```pycon
>>> print(qasm_string)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
x q[0];
cx q[2], q[1];
cx q[0], q[3];
x q[3];
cx q[0], q[4];
x q[3];
x q[2];
cx q[0], q[4];
x q[3];
x q[4];
x q[3];
cx q[2], q[1];
x q[0];
cx q[3], q[2];
x q[2];
cx q[0], q[4];
x q[2];
cx q[2], q[1];
```

as well as the permutation applied to the qubits (which is what the qubit allocation method needs to find).

```pycon
>>> print(perm)
[0 4 3 2 1]
```
