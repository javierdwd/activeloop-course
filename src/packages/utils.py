import networkx as nx
from pyvis.network import Network
from dotenv import load_dotenv, find_dotenv
from typing import List
from pathlib import Path
# Env


def load_config():

    load_dotenv(find_dotenv())

    load_dotenv('../.env')


# Create a NetworkX graph from the extracted relation triplets

def create_graph_from_triplets(triplets):
    G = nx.DiGraph()
    for triplet in triplets:
        subject, predicate, obj = triplet.strip().split(',')
        G.add_edge(subject.strip(), obj.strip(), label=predicate.strip())
    return G

# Convert the NetworkX graph to a PyVis network


def nx_to_pyvis(networkx_graph):
    pyvis_graph = Network(notebook=True)
    for node in networkx_graph.nodes():
        pyvis_graph.add_node(node)
    for edge in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(edge[0], edge[1], label=edge[2]["label"])
    return pyvis_graph


def draw_kgraph(triples_list: List[str], filename: str):
    triplets = [t.strip() for t in triples_list if t.strip()]
    graph = create_graph_from_triplets(triplets)
    pyvis_network = nx_to_pyvis(graph)

    # Customize the appearance of the graph
    pyvis_network.toggle_hide_edges_on_drag(True)
    pyvis_network.toggle_physics(False)
    pyvis_network.set_edge_smooth('discrete')

    # Show the interactive knowledge graph visualization
    path_to_file = (Path("tmp") / filename)
    pyvis_network.show(path_to_file.as_posix())
