import networkx as nx
import numpy as np
import tensorflow as tf

import sympy as sm

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
    def __init__(self, m, kappa, hz, hb, gen_rows=False):
        self.graph = nx.hexagonal_lattice_graph(2 * m - 1, 2 * m - 1, 
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = m
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self.gen_rows = gen_rows
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
            
    def _check_node(self, graph, first_row, second_row, i, j, 
                    el, current_node, edge_list, stop):
        if (graph.nodes[el]["edge_label"] == 'Edge' 
            and el not in first_row):
            first_row[el] = i
            graph.nodes[el]["edge_label"] = 'None'
            current_node = el
        elif (graph.nodes[el]["edge_label"] == 'Bulk' 
              and el not in second_row):
            second_row[el] = j
            graph.nodes[el]["edge_label"] = 'None'
            j += 1
            for node in graph.neighbors(el):
                if graph.nodes[node]["edge_label"] == 'Finish':
                    stop = True
                    first_row[node] = i
                elif ((node not in first_row) and 
                      (graph.nodes[node]["edge_label"] == 'Edge')):
                    first_row[node] = i
                    graph.nodes[node]["edge_label"] = 'None'
                    current_node = node
                if graph.nodes[node]["edge_label"] == 'Bulk':
                    edge_list.append(node)
        return (graph, first_row, second_row, i, j, 
                el, current_node, edge_list, stop)
    
    def _select_row(self, graph, row):
        graph.nodes[(self.m, row * 2 + 1)]["edge_label"] = 'None'
        graph.nodes[(self.m - 1, row * 2 + 1)]["edge_label"] = 'Finish'
        current_node = (self.m, row * 2 + 1)
        first_row, second_row = {}, {}
        first_row[current_node], i, j = 0, 1, 0
        stop, edge_list = False, []
    
        while not stop:
            for el in graph.neighbors(current_node):
                if graph.nodes[el]["edge_label"] != 'None':
                    (graph, first_row, second_row, i, j, 
                     el, current_node, edge_list, 
                     stop) = self._check_node(graph, first_row, 
                                              second_row, i, j, el, 
                                              current_node, edge_list, stop)
            i += 1
        for node in edge_list:
            graph.nodes[node]["edge_label"] = 'Edge'
      
        edge_indexes = np.zeros(len(first_row), dtype=np.int32)
        in_indexes = np.zeros(len(second_row), dtype=np.int32)    
        for node, i in first_row.items():
            edge_indexes[i] = graph.nodes[node]["flat_index"]
            self.row_number[node] = row
        for node, i in second_row.items():
            in_indexes[i] = graph.nodes[node]["flat_index"]
            self.row_number[node] = row
        self.edge_n.append(edge_indexes)
        self.in_n.append(in_indexes)
               
    def _add_row_labels(self):
        if not self.gen_rows:
            return 0
        graph = self.graph.copy()
        i = 0
        for node in graph.nodes:
            graph.nodes[node]["flat_index"] = i
            i += 1
            if node in self.edge_nodes:
                graph.nodes[node]["edge_label"] = 'Edge'
            else:
                graph.nodes[node]["edge_label"] = 'Bulk'
                
        self.edge_n = []
        self.in_n = []
        self.row_number = {}
        for row in range(self.m - 1):
            self._select_row(graph, row)
        for node in graph.nodes:
            if node not in self.row_number:
                self.row_number[node] = self.m - 1
            

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
            

class PeriodicSample(KitaevBase):
    def __init__(self, m, n, kappa):
        self.graph = nx.hexagonal_lattice_graph(m, n, 
                                                periodic=True, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = m
        self.n = n
        self.kappa = kappa
        self._set_parameters()
        
    def _set_edge_dir(self):
        pass
    
    def _add_edge_nodes(self):
        pass
    
class HexagonHole(KitaevBase):
    def __init__(self, l, m, kappa, hz, hb, gen_rows=False):
        self.graph = nx.hexagonal_lattice_graph(2 * l - 1, 2 * l, 
                                                periodic=True, 
                                                with_positions=True, 
                                                create_using=None)
        self.l = l
        self.m = m
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self.gen_rows = gen_rows
        self._set_parameters()
    
    def _remove_unwanted_nodes(self):
        cx, cy = self.l - 0.5, 2 * self.l - (self.l % 2) - 2  
        unwanted = []
        for el in self.graph.nodes:    
            x, y = el
            if ((abs(cx - x) + abs(cy - y) < 2 * self.m) 
                and (x > cx - self.m and x < cx + self.m)):
                unwanted.append(el)
        for el in unwanted:
            self.graph.remove_node(el)
            
    def _label_graph(self):
        graph = self.graph.copy()
        start_node, i = 0, 0
        for node in graph.nodes:
            graph.nodes[node]["flat_index"] = i
            i += 1
            if node in self.edge_nodes:
                graph.nodes[node]["edge_label"] = 'Edge'
            else:
                neib = graph.neighbors(node)
                edge_count = 0
                for el in neib:
                    if el in self.edge_nodes:
                        edge_count += 1     
                if edge_count == 1:
                    graph.nodes[node]["edge_label"] = 'Edge_2'
                    p_node = start_node
                    start_node = node
                elif edge_count == 2:
                    graph.nodes[node]["edge_label"] = 'Edge_3' 
                else:
                    graph.nodes[node]["edge_label"] = 'None'
        return graph, start_node, p_node
            
    def _check_node(self, graph, first_row, second_row, i, j, 
                    current_node):
        for el in graph.neighbors(current_node):
            if graph.nodes[el]["edge_label"] != 'None':
                if (graph.nodes[el]["edge_label"] == 'Edge'
                    and el not in first_row):
                    first_row[el] = i
                    current_node = el
                    i += 1
                    break
                elif (graph.nodes[el]["edge_label"] == 'Edge_2'
                    and el not in first_row):
                    first_row[el] = i
                    current_node = el
                elif (graph.nodes[el]["edge_label"] == 'Edge_3' 
                      and el not in second_row):
                    second_row[el] = j
                    j += 1
                    for node in graph.neighbors(el):
                        if ((node not in first_row) and 
                            (graph.nodes[node]["edge_label"] == 'Edge')):
                            first_row[node] = i
                            current_node = node
                            i += 1
        return (graph, first_row, second_row, i, j, current_node)
                
    def _add_row_labels(self):
        if not self.gen_rows:
            return 0
        graph, start_node, p_node = self._label_graph()   
        current_node = start_node
        first_row, second_row = {}, {}
        first_row[current_node], i, j = 0, 0, 0

        while current_node != p_node:
            (graph, first_row, second_row, i, j, 
             current_node) = self._check_node(graph, first_row, 
                                              second_row, i, j, 
                                              current_node)
                    
        edge_indexes = np.zeros(len(first_row) - 12, dtype=np.int32)
        for node, i in first_row.items():
            if graph.nodes[node]["edge_label"] == 'Edge':
                edge_indexes[i] = graph.nodes[node]["flat_index"]
        self.edge_n = edge_indexes    
        
        
if __name__ == '__main__':  
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    kit_model = HexagonHole(40, 28, 0.027, 0.3, 0, gen_rows=True)
    print(kit_model.edge_n)
    """
    kit_model = HexagonHole(20, 8, 0.027, 0.3, 0)
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[101] = 1
    kit_model.plot_state(initial_state, save=True, size=20, max_amp=0.1, colormap='viridis_r')
    fin_state = kit_model.evolution(initial_state, 100)
    kit_model.plot_state(fin_state, file_name='fin_state.pdf', save=True, size=10, max_amp=0.1)
    kit_model.animated_ev(initial_state, 0.4, 
                    frames=90, interval=10, repeat=False, 
                    file_name='anime.gif', save='False', 
                    size=10, max_amp=0.3)
    
    kit_model.add_disorder(mse=0.1, n_samples=5)
    initial_state = np.zeros(kit_model.e_mult.shape[-1])
    initial_state[21] = 1
    #kit_model.plot_state(initial_state, save=True, size=20, max_amp=0.1, colormap='viridis_r')
    #fin_states = kit_model.dis_evolution(initial_state, time=400)
    times, overlap = kit_model.dis_overlap(initial_state, 500, 600, n_times=100)
    print(np.max(overlap, axis=0))
    print('Done')
    
    
    kit_model.diagonalize()
    initial_state = np.zeros_like(kit_model.v[:, 0])
    initial_state[101] = 1
    kit_model.plot_state(initial_state, save=True, size=20, max_amp=0.1, colormap='viridis_r')
    fin_state = kit_model.evolution(initial_state, 100)
    kit_model.plot_state(fin_state, file_name='fin_state.pdf', save=True, size=10, max_amp=0.1)
    kit_model.animated_ev(initial_state, 0.4, 
                    frames=90, interval=10, repeat=False, 
                    file_name='anime.gif', save='False', 
                    size=10, max_amp=0.3)
                    """
    print('Done')
    
    #kit_model.plot_graph(file_name='g.pdf', save=True)