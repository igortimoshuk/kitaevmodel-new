import tensorflow as tf
import networkx as nx

def eigenstates(graph):
    hamiltonian = tf.linalg.band_part(tf.constant(nx.to_numpy_matrix(graph),
                                                  dtype=tf.complex64), 0, -1)

    hamiltonian = (hamiltonian -
                   tf.transpose(hamiltonian))
    
    return tf.linalg.eigh(1j * hamiltonian) 

def eigenvals(graph):
    hamiltonian = tf.linalg.band_part(tf.constant(nx.to_numpy_matrix(graph),
                                                  dtype=tf.complex64), 0, -1)

    hamiltonian = (hamiltonian -
                   tf.transpose(hamiltonian))
    
    return tf.linalg.eigh(1j * hamiltonian)[0]
