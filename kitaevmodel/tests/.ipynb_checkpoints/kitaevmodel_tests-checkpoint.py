from unittest import TestCase
import numpy as np
from kitaevmodel import hamiltonian_creator
from kitaevmodel import shapes
from kitaevmodel.shapes import HexagonZigzag

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
    
def test_hexagon_zigzag_sample():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    kit_model = HexagonZigzag(10, 0.027, 0.3, 0)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[31] = 1
    kit_model.plot_state(initial_state, save=True, size=20, max_amp=0.1, colormap='viridis_r')
    fin_state = kit_model.evolution(initial_state, 100)
    kit_model.plot_state(fin_state, file_name='fin_state.pdf', save=True, size=10, max_amp=0.1)
    kit_model.animated_ev(initial_state, 10, 
                    frames=30, interval=100, repeat=False, 
                    file_name='anime.gif', save='False', 
                    size=10, max_amp=0.3)
    print('Done')