# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import networkx as nx

from itertools import chain

from queko import (
    _backbone_construction,
    _sprinkling_phase,
    _scrambling_phase,
    _generate_qasm,
    queko_circuit,
)

# Hard-coded edge lists for 3x3 and 4x4 lattices
edges_3 = [
    (0, 3),
    (0, 1),
    (1, 4),
    (1, 2),
    (2, 5),
    (3, 6),
    (3, 4),
    (4, 7),
    (4, 5),
    (5, 8),
    (6, 7),
    (7, 8),
]

edges_4 = [
    (0, 4),
    (0, 1),
    (1, 5),
    (1, 2),
    (2, 6),
    (2, 3),
    (3, 7),
    (4, 8),
    (4, 5),
    (5, 9),
    (5, 6),
    (6, 10),
    (6, 7),
    (7, 11),
    (8, 12),
    (8, 9),
    (9, 13),
    (9, 10),
    (10, 14),
    (10, 11),
    (11, 15),
    (12, 13),
    (13, 14),
    (14, 15),
]


def test_invalid_specs():
    """Test that construction fails when the density vector and specified
    depth would yield a circuit that doesn't fit the hardware graph."""

    G = nx.grid_graph((3, 3))

    # [0.05, 0.05], depth 20 -> 9 1-qubit gates, 5 2-qubit gates; too small
    # to fit within depth 20
    with pytest.raises(ValueError, match="Insufficient gate densities"):
        queko_circuit(G, 20, [0.05, 0.05], lattice_dim=3)

    # [0.9, 0.9], depth 5 -> 41 1-qubit gates, 21 2-qubit gates; too large
    # to fit within depth 5
    with pytest.raises(ValueError, match="gate densities are too large"):
        queko_circuit(G, 20, [0.9, 0.9], lattice_dim=3)

    # [0.05, 0.9], depth 20 -> 81 2-qubit gates; divide 81 over max. 4
    # 2-qubit gates per layer of depth since size of maximal matching is 4,
    # can see that this would be invalid.
    with pytest.raises(ValueError, match="max. matching size"):
        queko_circuit(G, 20, [0.05, 0.9], lattice_dim=3)


@pytest.mark.parametrize(
    "N,edge_list,T,M2",
    [
        (3, edges_3, 5, 3),
        (3, edges_3, 10, 2),
        (3, edges_3, 20, 8),
        (4, edges_4, 20, 3),
        (4, edges_4, 10, 8),
    ],
)
def test_backbone(N, edge_list, T, M2):
    """Test that backbone construction yields circuits with the right
    length and properties."""

    gate_list, m1_count, m2_count = _backbone_construction(list(range(N ** 2)), edge_list, T, M2)

    print(gate_list)

    # Check length of the circuit
    assert len(gate_list) == T
    assert m1_count + m2_count == T
    assert m2_count <= M2

    # Check that the backbone has a proper dependency chain
    for idx, gate in enumerate(gate_list[1:]):
        prev_qubits = gate_list[idx][1]
        assert any([qubit in prev_qubits for qubit in gate[1]])


@pytest.mark.parametrize(
    "N,edge_list,T,M1,M2",
    [
        (3, edges_3, 5, 3, 3),
        (3, edges_3, 10, 5, 8),
        (3, edges_3, 20, 15, 10),
        (4, edges_4, 20, 10, 10),
        (4, edges_4, 10, 8, 7),
    ],
)
def test_sprinkling_phase(N, edge_list, T, M1, M2):
    """Test that sprinkling phase yields circuits with the right length and
    properties.
    """
    node_list = list(range(N ** 2))

    gate_list, m1_count, m2_count = _backbone_construction(node_list, edge_list, T, M2)

    gate_list = _sprinkling_phase(gate_list, node_list, edge_list, T, M1, M2, m1_count, m2_count)

    assert len(gate_list) <= M1 + M2

    # Make sure all time steps are valid
    for gate in gate_list:
        assert gate[0] < T

    # For each time step, make sure there is no gate overlap
    for t in range(T):
        qubits_this_timestep = [gate[1] for gate in gate_list if gate[0] == t]
        flat_qubits = list(chain.from_iterable(qubits_this_timestep))
        assert len(flat_qubits) == len(set(flat_qubits))


gate_list_sample = [
    (0, (2,)),
    (1, (2,)),
    (2, (2,)),
    (3, (2, 5)),
    (4, (2,)),
    (5, (2,)),
    (6, (1, 2)),
    (7, (2,)),
    (8, (2, 5)),
    (9, (4, 5)),
]


gate_list_sample_permuted = [
    (0, (6,)),
    (1, (6,)),
    (2, (6,)),
    (3, (6, 3)),
    (4, (6,)),
    (5, (6,)),
    (6, (7, 6)),
    (7, (6,)),
    (8, (6, 3)),
    (9, (4, 3)),
]


@pytest.mark.parametrize(
    "gate_list,perm,gate_list_permuted",
    [
        (gate_list_sample, [0, 1, 2, 3, 4, 5, 6, 7, 8], gate_list_sample),
        (gate_list_sample, [8, 7, 6, 5, 4, 3, 2, 1, 0], gate_list_sample_permuted),
    ],
)
def test_scrambling_phase(gate_list, perm, gate_list_permuted):
    """Test that the gates are correctly permuted."""
    result = _scrambling_phase(gate_list, perm)

    for gate_original, gate_result in zip(gate_list_permuted, result):
        assert gate_original[0] == gate_result[0]
        assert gate_original[1] == gate_result[1]


def test_generate_qasm():
    """Test that the QASM parser produces expected results."""
    gate_list_mini = [(0, (6,)), (1, (3, 7))]

    result = _generate_qasm(gate_list_mini, 9, {1: "h", 2: "cx"})
    expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[9];\nh q[6];\ncx q[3], q[7];\n'
    assert result == expected
