import tensorflow as tf
import numpy as np

import networkx as nx
import matplotlib
from matplotlib import animation, rc
from matplotlib import pyplot as plt

from scipy.signal import gaussian

def evaluation(new_state, time, e, v, v_inv):
    new_coeff = tf.linalg.matvec(v_inv, (new_state / tf.norm(new_state)))
    time *= 1j
    evol_coeffs = new_coeff * tf.math.exp(time * e)
    result_coeff = tf.linalg.matvec(v, evol_coeffs)
    return result_coeff

def evaluation_prep(new_state, time, e, v, v_inv):
    new_coeff = tf.linalg.matvec(v_inv, (new_state / tf.norm(new_state)))
    return new_coeff

def time_step(new_state, time, v, e):
    time_exp = tf.math.exp(1j * time * e)
    evol_coeffs = new_state * time_exp
    result_coeff = tf.linalg.matvec(v, evol_coeffs)
    return result_coeff

def update(num, current_state, v, e, graph, pos, time, ax, size):
    time = (num + 0.01) * time
    ax.clear()
    result_coeffs = time_step(current_state, time, v, e)
    
    colors = np.abs(result_coeffs)
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

def animated_repr(initial_state, time, graph, pos, max_x, max_y, e, v, v_inv, 
                  frames=135, interval=100, repeat=False, size=10):
    fig, ax = plt.subplots(figsize=(max_x / size, max_y / size))

    current_state = evaluation_prep(initial_state, time, e, v, v_inv)
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=frames, interval=interval, repeat=repeat,
                                      fargs=(current_state, v, e, graph, pos, time, ax, size))
    
    ani.save('anime.gif')

if __name__ == '__main__':
    #for test RTX 2080 or better requared
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from hamiltonian_creator import hexagon_lattice_zigzag
    from eigenstates import eigenstates
    from image_generator import state_on_the_lattice_uniform

    graph, pos, max_x, max_y = hexagon_lattice_zigzag(20, 0.027, 0.3, 0)
    e, v = eigenstates(graph)
    v_inv = tf.linalg.inv(v)

    initial_state = np.zeros_like(v[:, 0])
    initial_state[27] = 1

    time = 1
    initial_state = tf.constant(initial_state)
    animated_repr(initial_state, time, graph, pos, max_x, max_y, e, v, v_inv,
                  frames=20 * 2, interval=100, repeat=False, size=10)
    