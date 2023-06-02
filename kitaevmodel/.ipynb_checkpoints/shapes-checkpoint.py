import networkx as nx
import numpy as np
import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation

if __name__ != '__main__': 
    import kitaevmodel
    from kitaevmodel.basesample import KitaevBase
else:
    import basesample
    from basesample import KitaevBase

class HexagonZigzag(KitaevBase):
    def __init__(self, m, kappa, hz, hb):
        self.graph = nx.hexagonal_lattice_graph(2 * m - 1, 2 * m - 1, 
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = m
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self._set_parameters()
    
    def _remove_unwanted_nodes(self):
        cx, cy = self.m - 0.5, 2 * self.m - (self.m % 2) 
        unwanted = []
        for el in self.graph.nodes:    
            x, y = el
            if abs(cx - x) + abs(cy - y) > 2 * self.m:
                unwanted.append(el)
        for el in unwanted:
            self.graph.remove_node(el)
            

class BandZigzag(KitaevBase):
    def __init__(self, m, n, kappa, hz, hb):
        self.graph = nx.hexagonal_lattice_graph(2 * m - 1, 2 * n - 1, 
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = m
        self.n = n
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self._set_parameters()
    
    def _remove_unwanted_nodes(self):
        cx, cy = self.n - 0.5, 2 * self.m - (self.m % 2)
        unwanted = []
        for el in self.graph.nodes:    
            x, y = el
            if abs(cx - x) + abs(cy - y) > (self.m + self.n):
                unwanted.append(el)
        for el in unwanted:
            self.graph.remove_node(el)
            
            
class HexagonArmchair(KitaevBase):
    def __init__(self, m, kappa, hz, hb):
        m = m / 2 * 3 - 1
        self.graph = nx.hexagonal_lattice_graph(4 * int(m), 3 * int(m + 1),
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = 2 * m
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self._set_parameters()
    
    def _remove_unwanted_nodes(self):
        cx, cy = self.m + 1.5, self.m * np.sqrt(3) + 0.01
        pos = nx.get_node_attributes(self.graph, 'pos')
        unwanted = []
        for el in self.graph.nodes: 
            x, y = pos[el]
            if ((abs(y - cy) - 0.015 > np.sqrt(3) * 
                 min(self.m - abs(x - cx), self.m / 2)) 
                or el[1] == 4 * self.m + 1):
                unwanted.append(el)

        for el in unwanted:
            self.graph.remove_node(el)
            

class BandArmchair(KitaevBase):
    def __init__(self, m, n, kappa, hz, hb):
        m = m / 2 * 3 - 1
        n = n / 2 * 3 - 1
        self.graph = nx.hexagonal_lattice_graph(2 * int(n), 3 * int(m + 1), 
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = 2 * m
        self.n = n
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self._set_parameters()
    
    def _remove_unwanted_nodes(self):
        cx, cy = self.m + 1.5, self.n * np.sqrt(3) + 0.01
        pos = nx.get_node_attributes(self.graph, 'pos')
        unwanted = []
        for el in self.graph.nodes: 
            x, y = pos[el]
            if ((abs(y - cy) - 0.015 > np.sqrt(3) * 
                 min(self.m - abs(x - cx), self.m / 2)) 
                or el[1] == 4 * self.n + 1):
                unwanted.append(el)

        for el in unwanted:
            self.graph.remove_node(el)
    
        
        
if __name__ == '__main__':  
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    kit_model = KitaevBase(40, 40, 0.027, 0.3, 0)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[1021] = 1
    kit_model.plot_state(initial_state, save=True, size=20, max_amp=0.1, colormap='viridis_r')
    fin_state = kit_model.evolution(initial_state, 100)
    kit_model.plot_state(fin_state, file_name='fin_state.pdf', save=True, size=10, max_amp=0.1)
    kit_model.animated_ev(initial_state, 0.4, 
                    frames=90, interval=10, repeat=False, 
                    file_name='anime.gif', save='False', 
                    size=10, max_amp=0.3)
    print('Done')
    #kit_model.plot_graph(file_name='g.pdf', save=True)