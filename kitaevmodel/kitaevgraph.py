import networkx as nx
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

class HexagonZigzag:
    def __init__(self, m, kappa, hz, hb):
        self.graph = nx.hexagonal_lattice_graph(2*m-1,2*m-1, 
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = m
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        coord = np.array(list(self.pos.values()))
        
        self._remove_unwanted_nodes()
        self._find_edge()
        
        for edge in self.graph.edges(data=True):
            edge[2]['weight'] = 1
            
        for u, v, d in self.graph.edges(data=True):
            if (u[0] + u[1]) % 2 == 0:
                d['weight'] = -1
        
        self._add_kappa()
        self._add_edge_nodes()
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        
        if self.hb:
            self.max_x = np.max(coord[:, 0]) + 2
            self.max_y = np.max(coord[:, 1]) + np.sqrt(3)
        else:
            self.max_x = np.max(coord[:, 0])
            self.max_y = np.max(coord[:, 1])
    
    def _remove_unwanted_nodes(self):
        cx, cy = self.m - 0.5, 2 * self.m - (self.m % 2) 
        unwanted = []
        for n in self.graph.nodes:    
            x, y = n
            if abs(cx - x) + abs(cy - y) > 2 * self.m:
                unwanted.append(n)
        for n in unwanted:
            self.graph.remove_node(n)
            
    def _find_edge(self):
        self.edge_nodes = {}
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) == 2:
                self.edge_nodes[node] = neighbors  
    
    def _add_kappa(self):
        results = {}
        for node in self.graph.nodes:
            dist = set()
            for el in self.graph.neighbors(node):
                data = set(self.graph.neighbors(el))
                dist = set.union(dist, data)
                dist.remove(node)
        
            results[node] = dist
        
        for node, dist in results.items():
            for el in dist:
                if (el[0] + el[1]) % 2 == 0:
                    if el[0] != node[0]:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, weight=-self.kappa)
                        else:
                            self.graph.add_edge(node, el, weight=self.kappa)
                    else:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, weight=self.kappa)
                        else:
                            self.graph.add_edge(node, el, weight=-self.kappa)
                            
                if (el[0] + el[1]) % 2 == 1:
                    if el[0] != node[0]:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, weight=self.kappa)
                        else:
                            self.graph.add_edge(node, el, weight=-self.kappa)
                    else:
                        if el[1] > node[1]:
                            self.graph.add_edge(node, el, weight=-self.kappa)
                        else:
                            self.graph.add_edge(node, el, weight=self.kappa)
                            
    def _add_edge_nodes(self):
        if self.hb:
            pos = self.pos
            for node, value in self.edge_nodes.items():
                first, second = value
                place = (pos[node][0] + 
                         (2 * pos[node][0] - pos[first][0] - pos[second][0]) * 1, 
                         pos[node][1] + 
                         (2 * pos[node][1] - pos[first][1] - pos[second][1]) * 1)
                name = (node[0] + 10000, node[1] + 10001)
                self.graph.add_node(name, pos=place)

                self.graph.add_edge(name, node, weight=self.hb)
                signi = self.graph[first][second]['weight']
                self.graph.remove_edge(first, second)
                self.graph.add_edge(first, second, weight=(signi / self.hz * self.hb))
        else:
            for node, value in self.edge_nodes.items():
                first, second = value
                self.graph.remove_edge(first, second)
    
    def plot_graph(self, file_name='graph.pdf', save=False):
        matrix = nx.to_numpy_array(self.graph)
        matrix = np.triu(matrix)
        matrix = matrix - matrix.T
        place = list(self.pos.values())

        pos = {}
        for i in range(len(place)):
            pos[i] = place[i]

        directed_graph = nx.convert_matrix.from_numpy_array(matrix,
                                                            create_using=nx.DiGraph)

        edges = directed_graph.edges()
        edgelist = []
        weights = []
        for u, v in edges:
            if directed_graph[u][v]['weight'] > 0:
                edgelist.append((u, v))
                weights.append(directed_graph[u][v]['weight'])

        colors = []
        for node in self.graph.nodes():
            if (node[0] + node[1]) % 2 == 1:
                colors.append('black')
            else:
                colors.append('lightgray')

        plt.figure(figsize=(self.max_x, self.max_y))
        nx.draw(directed_graph,
                pos=pos,
                width=weights,
                edgelist=edgelist,
                node_color=colors,
                with_labels=False)

        if save:
            plt.savefig(file_name, bbox_inches = 'tight')
        plt.show()
        
    def diagonalize(self):
        hamiltonian = tf.linalg.band_part(tf.constant(nx.to_numpy_array(self.graph),
                                                  dtype=tf.complex64), 0, -1)

        hamiltonian = (hamiltonian -
                       tf.transpose(hamiltonian))
        
        self.e, self.v = tf.linalg.eigh(1j * hamiltonian)
        self.v_inf = tf.linalg.inv(self.v)
        
    def plot_state(self, state, size=1,  
                   file_name='state.pdf', 
                   save=False):
        state /= np.linalg.norm(state)
        colors = np.abs(state)

        white_nodes = []
        white_nodes_color = []
        black_nodes = []
        black_nodes_color = []
        i = 0

        for node in self.graph.nodes():
            if (node[0] + node[1]) % 2 == 1:
                black_nodes.append(node)
                black_nodes_color.append(colors[i])
            else:
                white_nodes.append(node)
                white_nodes_color.append(colors[i]) 
            i += 1

        plt.figure(figsize=(self.max_x / size, self.max_y / size)) 
        nx.draw_networkx_nodes(self.graph, 
                               self.pos, 
                               nodelist=black_nodes, 
                               node_shape=(3, 0, 270),
                               node_color=black_nodes_color,
                               node_size=9600 / size ** 2)

        nx.draw_networkx_nodes(self.graph, 
                               self.pos, 
                               nodelist=white_nodes, 
                               node_shape=(3, 0, 90),
                               node_color=white_nodes_color,
                               node_size=9600 / size ** 2)
        if save:
            plt.savefig(file_name, bbox_inches = 'tight')
        plt.show()
        
        
if __name__ == '__main__':                
    kit_model = HexagonZigzag(5, 0.027, 0.3, 0)
    kit_model.diagonalize()
    kit_model.plot_state(kit_model.v[:, 1250], save=True)
    #print(kit_model.pos)
    #kit_model.plot_graph(file_name='g.pdf', save=True)