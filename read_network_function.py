# Using "networkx" library
import networkx as nx
import matplotlib.pyplot as plt

network = "Surfnet.graphml.xml"

def read_graph(G):
    N = nx.read_graphml(G)          # read a graph (XML type)
    nodes = list(N.nodes)           # network's nodes
    edges = list()
    # num_nodes = len(nodes)          # The number of nodes
    for (e) in N.edges:
        edges.append((int(e[0]),int(e[1])))
        edges.append((int(e[1]),int(e[0])))
    G_D = nx.DiGraph()    # Build a directed graph
    #G_D.add_nodes_from(nodes)
    G_D.add_edges_from(edges)    # generate nodes automatically base on the edges
    u_edges = list()
    for (e) in N.edges:
        u_edges.append((int(e[0]),int(e[1])))
       
    G_U = nx.Graph()
    G_U.add_edges_from(u_edges)
    return (G_D,G_U)

def main():
    (DG,UG) = read_graph(network)
    nodes = list(DG.nodes)
    #nodes.sort()
    print(f"Nodes: {nodes}")
    print(f"Edges: {list(DG.edges)}")
    u_nodes = list(UG.nodes)
    u_edges = list(UG.edges)
    print(f"Undirected nodes: {u_nodes}")
    print(f"Undirected edges: {u_edges}")
    nx.draw(UG, node_color='lightblue', 
        with_labels=True, 
        node_size=500)

    plt.show()
    #plt.savefig('graph_simulation.png')

if __name__ == "__main__":
    main()