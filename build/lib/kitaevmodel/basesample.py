import networkx as nx
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation

class KitaevBase:
    def __init__(self, m, n, kappa, hz, hb, gen_rows=False):
        self.graph = nx.hexagonal_lattice_graph(m, n, 
                                                periodic=False, 
                                                with_positions=True, 
                                                create_using=None)
        self.m = m
        self.n = n
        self.kappa = kappa
        self.hz = hz
        self.hb = hb
        self._set_parameters()
        
    def _set_parameters(self):
        self._remove_unwanted_nodes()
        self._find_edge()
        self._add_row_labels()
        self._calc_figsize()   
        self._set_edge_dir()
        self._add_kappa()
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        self._add_edge_nodes()
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        
    def _remove_unwanted_nodes(self):
        pass
    
    def _add_row_labels(self):
        pass

    def _find_edge(self):
        self.edge_nodes = {}
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) == 2:
                self.edge_nodes[node] = neighbors
                
    def _set_edge_dir(self):
        for edge in self.graph.edges(data=True):
            edge[2]['weight'] = 1
            
        for u, v, d in self.graph.edges(data=True):
            if (u[0] + u[1]) % 2 == 0:
                d['weight'] = -1
    
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
    
    def _calc_figsize(self):
        coord = np.array(list(nx.get_node_attributes(self.graph, 'pos').values()))
        if self.hb:
            self.max_x = np.max(coord[:, 0]) + 2
            self.max_y = np.max(coord[:, 1]) + np.sqrt(3)
        else:
            self.max_x = np.max(coord[:, 0])
            self.max_y = np.max(coord[:, 1])
        
    
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
        self.v_inv = tf.linalg.inv(self.v)
        
    def _draw_state(self, state, size, max_amp, colormap):
        colors = np.abs(state)
        white_nodes = []
        white_nodes_color = []
        black_nodes = []
        black_nodes_color = []
        ext_qubits = []
        ext_qubits_color = []
        i = 0

        for node in self.graph.nodes():
            if len(node) == 2 and (node[0] + node[1]) % 2 == 1:
                black_nodes.append(node)
                black_nodes_color.append(colors[i])
            elif len(node) == 2:
                white_nodes.append(node)
                white_nodes_color.append(colors[i])
            else:
                ext_qubits.append(node)
                ext_qubits_color.append(colors[i])
            i += 1
        
        nx.draw_networkx_nodes(self.graph, 
                               self.pos, 
                               nodelist=black_nodes, 
                               node_shape=(3, 0, 270),
                               node_color=black_nodes_color,
                               node_size=9600 / size ** 2, 
                               cmap=colormap,
                               vmin=0, vmax=max_amp)

        nx.draw_networkx_nodes(self.graph, 
                               self.pos, 
                               nodelist=white_nodes, 
                               node_shape=(3, 0, 90),
                               node_color=white_nodes_color,
                               node_size=9600 / size ** 2,
                               cmap=colormap,
                               vmin=0, vmax=max_amp)
        
        nx.draw_networkx_nodes(self.graph, 
                               self.pos,  
                               nodelist=ext_qubits, 
                               node_shape='o',
                               node_color=ext_qubits_color,
                               node_size=4000 / size ** 2,
                               cmap=colormap, 
                               vmin=0, vmax=max_amp)
              
    def plot_state(self, state, size=1,  
                   file_name='state.pdf', 
                   save=False, max_amp=1, 
                   colormap='viridis'):
        state /= np.linalg.norm(state)
        plt.figure(figsize=(self.max_x / size, self.max_y / size))
        plt.box(False)
        self._draw_state(state, size, max_amp, colormap)
        
        if save:
            plt.savefig(file_name, bbox_inches='tight', 
                        pad_inches=0)
        plt.show()
        
    def evolution(self, state, time=1):
        coeff = tf.linalg.matvec(self.v_inv, (state / 
                                              tf.norm(state)))
        return tf.linalg.matvec(self.v, coeff * 
                                tf.math.exp(1j * time * self.e))
    
    def _update(self, num, eigst_coeff, time, size, max_amp, colormap):
        time = (num + 0.01) * time
        plt.clf()
        plt.box(False)
        
        state = tf.linalg.matvec(self.v, 
                                         eigst_coeff * 
                                         tf.math.exp(1j * time * self.e))
        self._draw_state(state, size, max_amp, colormap)
    
    def animated_ev(self, initial_state, time, 
                    frames=30, interval=100, repeat=False, 
                    file_name='anime.gif', save=False, 
                    size=10, max_amp=1, colormap='viridis'):
        fig, ax = plt.subplots(figsize=(self.max_x / size, self.max_y / size))
        eigst_coeff = tf.linalg.matvec(self.v_inv, 
                                       (initial_state / 
                                        tf.norm(initial_state)))
        ani = matplotlib.animation.FuncAnimation(fig, self._update, frames=frames, 
                                                 interval=interval, repeat=repeat, 
                                                 fargs=(eigst_coeff, time, 
                                                        size, max_amp, colormap))
        if save:
            ani.save(file_name)
        return ani.to_jshtml()
        
    def add_disorder(self, mse=0.1, n_samples=2, noise_vals=None):
        hamiltonian = tf.linalg.band_part(tf.constant(nx.to_numpy_array(self.graph),
                                                  dtype=tf.float32), 0, -1)
        hamiltonian = (hamiltonian - tf.transpose(hamiltonian)) 
        number_of_el = tf.where(hamiltonian).shape[0]
        mult_ham = tf.tensordot(tf.ones(n_samples, dtype=tf.float32), hamiltonian, 
                                axes=0)

        if noise_vals == None:
            noise_vals = tf.random.normal((n_samples, number_of_el), 1, mse)
        noise = tf.sparse.to_dense(tf.sparse.SparseTensor(
            indices=tf.where(mult_ham), 
            values=tf.reshape(noise_vals, [-1]),
            dense_shape=mult_ham.shape))
        noise = (noise + tf.transpose(noise, perm=[0,2,1])) / 2
        self.n_samples = n_samples
        self.e_mult, self.v_mult = tf.linalg.eigh(1j * tf.cast(noise * mult_ham, 
                                                               dtype=tf.complex64))
        self.v_inv_mult = tf.linalg.inv(self.v_mult)
        
    def _add_repl(self, state):
        state = state / np.linalg.norm(state)
        return tf.tensordot(tf.ones(self.n_samples, dtype=tf.complex64), 
                            tf.constant(state, dtype=tf.complex64),
                            axes=0)
        
    def dis_evolution(self, state, time=1):
        if len(state.shape) == 1:
            state = self._add_repl(state)    
        coeff = tf.linalg.matvec(self.v_inv_mult, state)
        return tf.linalg.matvec(self.v_mult, coeff * 
                                tf.math.exp(1j * time * self.e_mult))
    
    def dis_overlap(self, in_state, time_s, time_f, n_times=1000, fin_state=None):
        if len(in_state.shape) == 1:
            in_state = self._add_repl(in_state)
        if fin_state == None:
            fin_state = in_state
        elif len(fin_state.shape) == 1:
            fin_state = self._add_repl(fin_state) 
        times = tf.cast(tf.linspace(time_s, time_f, n_times),
                        dtype=tf.complex64)
        phases = tf.tensordot(times, self.e_mult, axes=0)
        
        in_coeff = tf.linalg.matvec(self.v_inv_mult, in_state)
        fin_coeff = tf.linalg.matvec(self.v_inv_mult, fin_state)
        time_var_coeff = tf.tensordot(tf.ones([n_times], dtype=tf.complex64), 
                                  in_coeff, axes=0)
        overlap_var = tf.math.abs(tf.math.reduce_sum(time_var_coeff * 
                                                     tf.math.exp(1j * phases) *  
                                                     tf.math.conj(fin_coeff), axis=2))
        return times, overlap_var
