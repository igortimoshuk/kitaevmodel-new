import networkx as nx
import numpy as np

def node_dist(x,y, cx, cy):
    return abs(cx-x) + abs(cy-y)

def remove_unwanted_nodes(G, m, k):
    cx, cy = k-0.5, 2*m -(m%2) 
    
    unwanted = []
    for n in G.nodes:    
        x,y = n
        if node_dist(x,y, cx, cy) > (m + k):
            unwanted.append(n)

    for n in unwanted:
        G.remove_node(n)
        
    return G

def add_kappa(graph, kappa):
    results = {}
    for node in graph.nodes:
        dist = set()
        for el in graph.neighbors(node):
            data = set(graph.neighbors(el))
            dist = set.union(dist, data)
        dist.remove(node)
        
        results[node] = dist
        
    for node, dist in results.items():
        for el in dist:
            if (el[0] + el[1]) % 2 == 0:
                if el[0] != node[0]:
                    if el[1] > node[1]:
                        graph.add_edge(node, el, weight=-kappa)
                    else:
                        graph.add_edge(node, el, weight=kappa)
                else:
                    if el[1] > node[1]:
                        graph.add_edge(node, el, weight=kappa)
                    else:
                        graph.add_edge(node, el, weight=-kappa)
            if (el[0] + el[1]) % 2 == 1:
                if el[0] != node[0]:
                    if el[1] > node[1]:
                        graph.add_edge(node, el, weight=kappa)
                    else:
                        graph.add_edge(node, el, weight=-kappa)
                else:
                    if el[1] > node[1]:
                        graph.add_edge(node, el, weight=-kappa)
                    else:
                        graph.add_edge(node, el, weight=kappa)
                
    return graph

def find_edge(graph):
    result = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 2:
            result[node] = neighbors  
    return result

def add_edge_nodes(graph, edge_nodes, hz, hb):
    if hb:
        pos = nx.get_node_attributes(graph, 'pos')
        for node, value in edge_nodes.items():
            first, second = value
            place = (pos[node][0] + 
                     (2 * pos[node][0] - pos[first][0] - pos[second][0]) * 1, 
                     pos[node][1] + 
                     (2 * pos[node][1] - pos[first][1] - pos[second][1]) * 1)
            name = (node[0] + 10000, node[1] + 10001)
            graph.add_node(name, pos=place)

            graph.add_edge(name, node, weight=hb)
            signi = graph[first][second]['weight']
            graph.remove_edge(first, second)
            graph.add_edge(first, second, weight=(signi / hz * hb))
    else:
        for node, value in edge_nodes.items():
            first, second = value
            graph.remove_edge(first, second)
        
    return graph

def rectangular_lattice(m, n, kappa, hz, hb):
    graph = nx.hexagonal_lattice_graph(m, n, periodic=False, 
                               with_positions=True, 
                               create_using=None)
    
    edge_nodes = find_edge(graph)
    
    for edge in graph.edges(data=True):
        edge[2]['weight'] = 1
    for u, v, d in graph.edges(data=True):
        if (u[0] + u[1]) % 2 == 0:
            d['weight'] = -1
            
    pos = nx.get_node_attributes(graph, 'pos')
    coord = np.array(list(pos.values()))
            
    graph = add_kappa(graph, kappa)
    graph = add_edge_nodes(graph, edge_nodes, hz, hb)
    
    pos = nx.get_node_attributes(graph, 'pos')
    if hb:
        max_x = np.max(coord[:, 0]) + 2
        max_y = np.max(coord[:, 1]) + np.sqrt(3)
    else:
        max_x = np.max(coord[:, 0])
        max_y = np.max(coord[:, 1])
    
    return graph, pos, max_x, max_y

def hexagon_lattice_zigzag(m, kappa, hz, hb):
    graph = nx.hexagonal_lattice_graph(2*m-1,2*m-1, periodic=False, 
                               with_positions=True, 
                               create_using=None)
    
    graph = remove_unwanted_nodes(graph, m, m)
    
    edge_nodes = find_edge(graph)
    
    for edge in graph.edges(data=True):
        edge[2]['weight'] = 1
    for u, v, d in graph.edges(data=True):
        if (u[0] + u[1]) % 2 == 0:
            d['weight'] = -1
            
    pos = nx.get_node_attributes(graph, 'pos')
    coord = np.array(list(pos.values()))
            
    graph = add_kappa(graph, kappa)
    graph = add_edge_nodes(graph, edge_nodes, hz, hb)
    
    pos = nx.get_node_attributes(graph, 'pos')
    if hb:
        max_x = np.max(coord[:, 0]) + 2
        max_y = np.max(coord[:, 1]) + np.sqrt(3)
    else:
        max_x = np.max(coord[:, 0])
        max_y = np.max(coord[:, 1])
    
    return graph, pos, max_x, max_y

def stripe_lattice_zigzag(m, n, kappa, hz, hb):
    graph = nx.hexagonal_lattice_graph(2*m-1,2*n - 1, periodic=False, 
                               with_positions=True, 
                               create_using=None)
    
    graph = remove_unwanted_nodes(graph, m, n)
    
    edge_nodes = find_edge(graph)
    
    for edge in graph.edges(data=True):
        edge[2]['weight'] = 1
    for u, v, d in graph.edges(data=True):
        if (u[0] + u[1]) % 2 == 0:
            d['weight'] = -1
            
    pos = nx.get_node_attributes(graph, 'pos')
    coord = np.array(list(pos.values()))
            
    graph = add_kappa(graph, kappa)
    graph = add_edge_nodes(graph, edge_nodes, hz, hb)
    
    pos = nx.get_node_attributes(graph, 'pos')
    if hb:
        max_x = np.max(coord[:, 0]) + 2
        max_y = np.max(coord[:, 1]) + np.sqrt(3)
    else:
        max_x = np.max(coord[:, 0])
        max_y = np.max(coord[:, 1])
    
    return graph, pos, max_x, max_y

def hexagon_check(m, pos):
        x, y = map(abs, pos)
        return y - 0.015 > np.sqrt(3) * min(m - x, m / 2)

def remove_unwanted_nodes_armchair(G, m, k):
    cx, cy = m + 1.5, k * np.sqrt(3) + 0.01
    pos = nx.get_node_attributes(G, 'pos')
    unwanted = []
    for n in G.nodes: 
        x, y = pos[n]
        if hexagon_check(m, (x - cx, y - cy)) or n[1] == 4 * k + 1:
            unwanted.append(n)

    for n in unwanted:
        G.remove_node(n)
        
    return G

def hexagon_lattice_armchair(m, kappa, hz, hb):
    m = m / 2 * 3 - 1
    graph = nx.hexagonal_lattice_graph(4 * int(m), 3 * int(m + 1),
                               periodic=False, 
                               with_positions=True, 
                               create_using=None)
    
    graph = remove_unwanted_nodes_armchair(graph, 2 * m, 2 * m)
    
    edge_nodes = find_edge(graph)
    
    for edge in graph.edges(data=True):
        edge[2]['weight'] = 1
    for u, v, d in graph.edges(data=True):
        if (u[0] + u[1]) % 2 == 0:
            d['weight'] = -1
            
    pos = nx.get_node_attributes(graph, 'pos')
    coord = np.array(list(pos.values()))
            
    graph = add_kappa(graph, kappa)
    graph = add_edge_nodes(graph, edge_nodes, hz, hb)
    
    pos = nx.get_node_attributes(graph, 'pos')
    if hb:
        max_x = np.max(coord[:, 0]) + 2
        max_y = np.max(coord[:, 1]) + np.sqrt(3)
    else:
        max_x = np.max(coord[:, 0])
        max_y = np.max(coord[:, 1])
    
    return graph, pos, max_x, max_y

def stripe_lattice_armchair(m, n, kappa, hz, hb):
    m = m / 2 * 3 - 1
    n = n / 2 * 3 - 1
    graph = nx.hexagonal_lattice_graph(2 * int(n), 3 * int(m + 1),
                               periodic=False, 
                               with_positions=True, 
                               create_using=None)
    
    graph = remove_unwanted_nodes_armchair(graph, 2 * m, n)
    
    edge_nodes = find_edge(graph)
    
    for edge in graph.edges(data=True):
        edge[2]['weight'] = 1
    for u, v, d in graph.edges(data=True):
        if (u[0] + u[1]) % 2 == 0:
            d['weight'] = -1
            
    pos = nx.get_node_attributes(graph, 'pos')
    coord = np.array(list(pos.values()))
            
    graph = add_kappa(graph, kappa)
    graph = add_edge_nodes(graph, edge_nodes, hz, hb)
    
    pos = nx.get_node_attributes(graph, 'pos')
    if hb:
        max_x = np.max(coord[:, 0]) + 2
        max_y = np.max(coord[:, 1]) + np.sqrt(3)
    else:
        max_x = np.max(coord[:, 0])
        max_y = np.max(coord[:, 1])
    
    return graph, pos, max_x, max_y

def add_disorder(graph, disorder, kappa_disorder):
    N = graph.number_of_edges()
    disorder_params = np.random.normal(1, disorder, N)
    i = 0
    for node_1, node_2, data in graph.edges(data=True):
        weight = data['weight']
        if np.abs(weight) > 0.5:
            graph[node_1][node_2]['weight'] *= disorder_params[i]
        else:
            graph[node_1][node_2]['weight'] *= (disorder_params[i] * 
                               kappa_disorder)
        i += 1
    return graph

if __name__ == '__main__':
    graph, pos, max_x, max_y = hexagon_lattice_zigzag(2, 0.1, 0.4, 0)
    print(pos)