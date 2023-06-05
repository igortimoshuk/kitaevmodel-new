import os
from unittest import TestCase
import numpy as np
from kitaevmodel import hamiltonian_creator
from kitaevmodel import shapes
from kitaevmodel.basesample import KitaevBase
from kitaevmodel.shapes import (HexagonZigzag, 
                                BandZigzag, 
                                HexagonArmchair, 
                                BandArmchair, 
                                PeriodicSample, 
                                HexagonHole, 
                                HexagonZigzagLinSpec) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    
def test_kitaev_base():
    kit_model = KitaevBase(3, 3, 0.027, 0.3, 0)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[0] = 1
    fin_state = kit_model.evolution(initial_state, 10)
    kit_model.animated_ev(initial_state, 10, 
                          frames=3, interval=1, repeat=False, 
                          file_name='anime.gif', save='False', 
                          size=10, max_amp=0.3)
    assert abs(np.abs(fin_state.numpy())[-1] - 0.090334624) <=1e-5
    
def test_hexagon_zigzag_sample():
    kit_model = HexagonZigzag(3, 0.027, 0.3, 0.1)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[0] = 1
    fin_state = kit_model.evolution(initial_state, 10)
    kit_model.animated_ev(initial_state, 10, 
                          frames=3, interval=1, repeat=False, 
                          file_name='anime.gif', save='False', 
                          size=10, max_amp=0.3)
    assert abs(np.abs(fin_state.numpy())[-1] - 0.047388554) <=1e-5
    
def test_band_zigzag_sample():
    kit_model = BandZigzag(3, 4, 0.027, 0.3, 0.1)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[0] = 1
    fin_state = kit_model.evolution(initial_state, 10)
    kit_model.animated_ev(initial_state, 10, 
                          frames=3, interval=1, repeat=False, 
                          file_name='anime.gif', save='False', 
                          size=10, max_amp=0.3)
    assert abs(np.abs(fin_state.numpy())[-1] - 0.07083912) <=1e-5
    
def test_hexagon_armchair_sample():
    kit_model = HexagonArmchair(3, 0.027, 0.3, 0.1)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[0] = 1
    fin_state = kit_model.evolution(initial_state, 10)
    kit_model.animated_ev(initial_state, 10, 
                          frames=3, interval=1, repeat=False, 
                          file_name='anime.gif', save='False', 
                          size=10, max_amp=0.3)
    assert abs(np.abs(fin_state.numpy())[-1] - 0.004621852) <=1e-5
    
def test_band_armchair_sample():
    kit_model = BandArmchair(3, 3, 0.027, 0.3, 0.1)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[0] = 1
    fin_state = kit_model.evolution(initial_state, 10)
    kit_model.animated_ev(initial_state, 10, 
                          frames=3, interval=1, repeat=False, 
                          file_name='anime.gif', save='False', 
                          size=10, max_amp=0.3)
    assert abs(np.abs(fin_state.numpy())[-1] - 0.019862168) <=1e-5
  
'''
def test_periodic_sample():
    kit_model = PeriodicSample(3, 3, 0.027, 0.3, 0.1)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[0] = 1
    fin_state = kit_model.evolution(initial_state, 10)
    kit_model.animated_ev(initial_state, 10, 
                          frames=3, interval=1, repeat=False, 
                          file_name='anime.gif', save='False', 
                          size=10, max_amp=0.3)
    assert abs(np.abs(fin_state.numpy())[-1] - 0.019862168) <=1e-5
    '''