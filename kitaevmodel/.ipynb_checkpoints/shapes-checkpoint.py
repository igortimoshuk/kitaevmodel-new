import networkx as nx
import numpy as np
import tensorflow as tf

import sympy as sm

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation

import kitaevmodel
from kitaevmodel.basesample import KitaevBase


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
                                              current_node, edge_list, 
                                              stop)
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
                
    def _draw_state_expand(self, state, max_amp, colormap):
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_amp)
        mapping = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
        plt.figure(figsize=(20, 5))
        plt.axis('off')
    
        for i in range(self.m - 1):
            N = len(self.edge_n[i]) // 6
            M = len(self.in_n[i]) // 6
            for j in range(6):
                lower_sites = np.linspace(j * 100, 
                                          (j + 1) * 100, N + 1)
                upper_sites = np.linspace(j * 100, 
                                          (j + 1) * 100, N)

                x_0 = np.vstack((lower_sites[:-1], upper_sites, 
                                 lower_sites[1:])).T
                y_0 = np.vstack((np.zeros(N), np.ones(N), 
                                 np.zeros(N))).T
                x_1 = np.vstack((upper_sites[:-1], lower_sites[1:-1], 
                                 upper_sites[1:])).T
                y_1 = np.vstack((np.ones(M), np.zeros(M), 
                                 np.ones(M))).T

                color_0 = np.abs(state[self.edge_n[i][N * j: N * (j + 1)]])
                color_1 = np.abs(state[self.in_n[i][M * j: M * (j + 1)]])
                for k in range(N):
                    plt.fill(x_0[k], y_0[k] + i, 
                             facecolor=mapping.to_rgba(color_0[k]))
                for m in range(M):
                    plt.fill(x_1[m], y_1[m] + i, 
                             facecolor=mapping.to_rgba(color_1[m]))
                
    def plot_state_expand(self, state,  
                          file_name='state_expand.pdf', 
                          save=False, max_amp=1, 
                          colormap='viridis'):
        state /= np.linalg.norm(state)
        self._draw_state_expand(state, max_amp, colormap)
        
        if save:
            plt.savefig(file_name, bbox_inches='tight', 
                        pad_inches=0)
        plt.show()
        
    def add_readout(self, depth, side='left', lambd=1):
        if self.hb:
            return 0
        if side == 'left':
            positions = (self.m - 2 ** depth + 1 + (depth == 0), 
                         self.m + 2 ** depth + 1 + (depth == 0))
            direct = -1
        elif side == 'right':
            N = len(self.pos)
            positions = (N - self.m - 2 ** depth - (depth == 0), 
                         N - self.m + 2 ** depth - (depth == 0))
            direct = 1
        else:
            return 0
        
        connect_nodes = list(self.graph.nodes())[positions[0]:positions[1]:2]
        connect_pos = []
        for node in connect_nodes:
            connect_pos.append(self.pos[node])
        if depth == 0:
            self.graph.add_node((0, (1 + direct) // 2, 0), 
                                pos=(connect_pos[0][0] + direct, 
                                     connect_pos[0][1]))
            self.graph.add_edge((0, (1 + direct) // 2, 0), 
                                connect_nodes[0], weight=-lambd * direct)
        for i in range(depth):
            curr_nodes = []
            curr_pos = []
            for j in range(len(connect_nodes)):
                name = (2 * i, j * 2 + (1 + direct) // 2, 0)
                curr_nodes.append(name)
                place = (connect_pos[j][0] + direct, connect_pos[j][1])
                curr_pos.append(place)
                self.graph.add_node(name, pos=place)
                self.graph.add_edge(name, connect_nodes[j], weight=-lambd * direct)
                
            connect_nodes = []
            connect_pos = []
            for j in range(len(curr_nodes) // 2):
                name = (2 * i, j * 2 + (1 - direct) // 2, 1)
                connect_nodes.append(name)
                place = (curr_pos[2 * j][0] + direct, 
                         (curr_pos[2 * j][1] + 
                          curr_pos[2 * j + 1][1]) / 2)
                connect_pos.append(place)
                self.graph.add_node(name, pos=place)
                self.graph.add_edge(name, curr_nodes[2 * j], weight=lambd * direct)
                self.graph.add_edge(name, curr_nodes[2 * j + 1], weight=lambd * direct)
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        self.max_x +=  depth * 2
            

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
        self.hb = 0
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
        

class HexagonZigzagLinSpec(HexagonZigzag):
    def __init__(self, m, kappa, hz, hb):
        self.graph = nx.hexagonal_lattice_graph(2 * m - 1, 2 * m - 1, 
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = m
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self.gen_rows = True
        self._set_parameters()
        
    def _add_kappa(self):
        results = {}
        for node in self.graph.nodes:
            dist = set()
            for el in self.graph.neighbors(node):
                data = set(self.graph.neighbors(el))
                dist = set.union(dist, data)
                dist.remove(node)
            results[node] = dist
        
        koeff = self.koeff_for_lin_spect()
        koeff_full = np.zeros(self.m + 1)
        koeff_full[:len(koeff)] = koeff
        for node, dist in results.items():
            for el in dist:
                edge_number = max(self.row_number[node], 
                                  self.row_number[el])
                mult = -koeff_full[edge_number]
                if (el[0] + el[1]) % 2 == 0:
                    if el[0] != node[0]:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, 
                                                weight=-self.kappa * (1-mult))
                        else:
                            self.graph.add_edge(node, el, 
                                                weight=self.kappa * (1-mult))
                    else:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, 
                                           weight=self.kappa * (1-mult))
                        else:
                            self.graph.add_edge(node, el, 
                                           weight=-self.kappa * (1-mult))
                if (el[0] + el[1]) % 2 == 1:
                    if el[0] != node[0]:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, 
                                           weight=self.kappa * (1-mult))
                        else:
                            self.graph.add_edge(node, el, 
                                           weight=-self.kappa * (1-mult))
                    else:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, 
                                           weight=-self.kappa * (1-mult))
                        else:
                            self.graph.add_edge(node, el, 
                                           weight=self.kappa * (1-mult))
        
    def koeff_for_lin_spect(self):
        x = sm.Symbol('x')
        ser = sm.N(sm.series(3 * (3 * 3 ** 0.5 * (1 - 2 * sm.acos(x/2)
                                                  / sm.pi)
                   / (4 - x ** 2) ** 0.5 - x) / (1 - x ** 2), x, n=self.m * 2))
        koeff = ser.as_coefficients_dict()
        n = len(koeff) - 1
        result = np.zeros(n)
        current = 0
        for i in range(n - 1, -1, -1):
            current = koeff[x ** (2 * i + 1)] - 2 * current
            result[i] = current
        return result
