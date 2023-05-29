from unittest import TestCase
from kitaevmodel import hamiltonian_creator

CORRECT_POS = {(0, 0): (0.5, 0.0),
 (0, 1): (0.0, 0.8660254037844386),
 (0, 2): (0.5, 1.7320508075688772),
 (1, 0): (1.5, 0.0),
 (1, 1): (2.0, 0.8660254037844386),
 (1, 2): (1.5, 1.7320508075688772)}

def test_graph_pos():
    graph, pos, max_x, max_y = hamiltonian_creator.hexagon_lattice_zigzag(1, 0.1, 0.4, 0)
    print(pos)
    assert pos == CORRECT_POS