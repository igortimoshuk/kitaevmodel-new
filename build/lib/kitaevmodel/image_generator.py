import networkx as nx
import numpy as np

if __name__ == '__main__':
    #for test RTX 2080 or better requared
    import os
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import tensorflow as tf
from matplotlib import pyplot as plt

def lattice_with_arrows(graph, pos, max_x, max_y, 
                        file_name='images/lattice_with_interaction.pdf',
                        save=False, 
                        size=1):
    matrix = nx.to_numpy_matrix(graph)
    matrix = np.triu(matrix)
    matrix = matrix - matrix.T
    place = list(pos.values())
    
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
    for node in graph.nodes():
        if (node[0] + node[1]) % 2 == 1:
            colors.append('black')
        else:
            colors.append('lightgray')

    plt.figure(figsize=(max_x / size, max_y / size))
    nx.draw(directed_graph,
            pos=pos,
            width=weights,
            edgelist=edgelist,
            node_color=colors,
            with_labels=False)
    
    if save:
        plt.savefig(file_name)
    plt.show()

def state_on_the_lattice_nodes(graph, pos, state, max_x, max_y, size=1, 
                               file_name='images/state_on_nodes.pdf', save=False):
    colors = np.abs(state)
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u,v in edges]

    plt.figure(figsize=(max_x / size, max_y / size)) 
    nx.draw(graph, 
            pos=pos, 
            node_color=colors, 
            width=weights, 
            with_labels=False, 
            node_size=1000 / size)
    if save:
        plt.savefig(file_name)
    plt.show()

def state_on_the_lattice_uniform(graph, pos, state, max_x, max_y, size=1, 
                                 file_name='images/state_uniform.pdf', save=False):
    colors = np.abs(state)
    
    white_nodes = []
    white_nodes_color = []
    black_nodes = []
    black_nodes_color = []
    i = 0
    
    for node in graph.nodes():
        if (node[0] + node[1]) % 2 == 1:
            black_nodes.append(node)
            black_nodes_color.append(colors[i])
        else:
            white_nodes.append(node)
            white_nodes_color.append(colors[i]) 
        i += 1

    plt.figure(figsize=(max_x / size, max_y / size)) 
    nx.draw_networkx_nodes(graph, 
                           pos, 
                           nodelist=black_nodes, 
                           node_shape=(3, 0, 270),
                           node_color=black_nodes_color,
                           node_size=9600 / size ** 2)
    
    nx.draw_networkx_nodes(graph, 
                           pos, 
                           nodelist=white_nodes, 
                           node_shape=(3, 0, 90),
                           node_color=white_nodes_color, 
                           node_size=9600 / size ** 2)
    if save:
        plt.savefig(file_name)
    plt.show()

def spectrum_and_density(e, save=False):
    energies = tf.make_ndarray(tf.make_tensor_proto(tf.math.real(e)))
    plt.plot(energies)
    plt.show()
    plt.title('Density of states')
    plt.xlabel('Energy, J')
    plt.hist(energies, 100)
    if save:
        plt.savefig('images/Density_of_states.png')
    plt.show()


if __name__ == '__main__':
    from eigenstates import eigenstates
    from hamiltonian_creator import (rectangular_lattice,
                                 hexagon_lattice_zigzag,
                                 stripe_lattice_zigzag, 
                                 hexagon_lattice_armchair, 
                                 stripe_lattice_armchair)

    graph, pos, max_x, max_y = rectangular_lattice(3, 5, 0.1, 0.4, 0)
    lattice_with_arrows(graph, pos, max_x, max_y, 
                    file_name='images/lattice_with_interaction.pdf', 
                    save=False, 
                    size=1)

    graph, pos, max_x, max_y = hexagon_lattice_zigzag(2, 0.1, 0.4, 0)
    lattice_with_arrows(graph, pos, max_x, max_y, 
                    file_name='images/lattice_with_interaction.pdf', 
                    save=False, 
                    size=1)

    graph, pos, max_x, max_y = stripe_lattice_zigzag(3, 2, 0.1, 0.4, 0)
    lattice_with_arrows(graph, pos, max_x, max_y, 
                    file_name='images/lattice_with_interaction.pdf', 
                    save=False, 
                    size=1)
                     
    graph, pos, max_x, max_y = hexagon_lattice_armchair(2, 0.1, 0.4, 0)
    lattice_with_arrows(graph, pos, max_x, max_y, 
                    file_name='images/lattice_with_interaction.pdf', 
                    save=False, 
                    size=1)

    graph, pos, max_x, max_y = stripe_lattice_armchair(3, 2, 0.1, 0.4, 0)
    lattice_with_arrows(graph, pos, max_x, max_y, 
                    file_name='images/lattice_with_interaction.pdf', 
                    save=False, 
                    size=1)
    
    e, v = eigenstates(graph)
    state = tf.make_ndarray(tf.make_tensor_proto(v[:, 3]))
    state_on_the_lattice_nodes(graph, pos, state, max_x, max_y, 
                               file_name='images/state_on_nodes.png')
    state_on_the_lattice_uniform(graph, pos, state, max_x, max_y, 
                                 file_name='images/state_uniform.pdf')
    spectrum_and_density(e)

