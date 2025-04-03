import yaml
import torch
from argparse import ArgumentParser
from src.agents.function_approximators import MessagePassingGNN
from src.agents.ppo import  PPO
from src.environments.mujoco_parser import parse_mujoco_graph, convert_mujoco_to_pytorch_geometric
import scipy as sp

def main(hyperparams):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YAML hyperparameters
    with open(f'utils/hyperparameters.yaml', 'r') as f:
        hparam = yaml.safe_load(f)

    # Replace hparams with command line arguments
    for k, v in vars(hyperparams).items():
        if v is not None:
            hparam[k] = v

    ### testing GNN ###
    # num_nodes = 10
    # num_features = 16
    #
    # # Random node features
    # x = torch.randn(num_nodes, num_features)
    #
    # # Random edges (just for demonstration)
    # edge_index = torch.tensor([
    #     [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0],  # Source nodes
    #     [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0, 9]  # Target nodes
    # ], dtype=torch.long)
    #
    # model = MessagePassingGNN(16, 10, device)
    # output = model(x, edge_index)
    # print(output)

    ### testing PPO ###
    # actor = MessagePassingGNN(device, in_dim, out_dim, **hparam)
    # model = PPO(device, actor, **hparam)
    # model.learn()

    ### testing mujoco parser ###
    x=parse_mujoco_graph(
        "CentipedeSix-v1",
        gnn_node_option="nG,yB",
    )

    # data=convert_mujoco_to_pytorch_geometric(x)
    # print(f"Graph created with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
    # print(f"Node feature dimensions: {data.x.size()}")
    # print(f"Edge feature dimensions: {data.edge_attr.size()}")
    #
    # from torch_geometric.utils import to_networkx
    # import networkx as nx
    # import matplotlib.pyplot as plt
    #
    # G = to_networkx(data, to_undirected=True)
    #
    # # Create labels for nodes
    # labels = {i: name for i, name in enumerate(data.node_names)}
    #
    # # Plot
    # plt.figure(figsize=(10, 10))
    # pos = nx.spring_layout(G, seed=42)  # Position nodes using a spring layout
    # nx.draw(G, pos, labels=labels, node_size=500, font_size=8)
    # plt.title("Graph Structure")
    # plt.savefig("graph_visualization.png")
    # print("Graph saved to graph_visualization.png")

    import json
    import networkx as nx
    import matplotlib.pyplot as plt

    # Load the data
    data=x

    # Create an empty directed graph
    G = nx.DiGraph()

    # Extract nodes from the tree
    if 'tree' in data:
        nodes = data['tree']
    else:
        nodes = data  # If the data is already a list of nodes

    # Add all nodes to the graph
    for node in nodes:
        G.add_node(node['id'],
                   name=node.get('name', f"node_{node['id']}"),
                   type=node.get('type', 'unknown'))

    # Add edges based on parent-child relationships
    for node in nodes:
        # Add edge from parent to this node
        if 'parent' in node and node['parent'] is not None:
            G.add_edge(node['parent'], node['id'])

        # Add edges to children
        if 'children_id_list' in node:
            for child_id in node['children_id_list']:
                G.add_edge(node['id'], child_id)

    # Print basic info about the graph
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Save the graph visualization to a file
    plt.figure(figsize=(12, 12))
    pos = nx.kamada_kawai_layout(G)
    # Use node types for colors
    node_colors = []
    for node_id in G.nodes():
        node_type = G.nodes[node_id]['type']
        if node_type == 'root':
            node_colors.append('red')
        elif node_type == 'joint':
            node_colors.append('blue')
        elif node_type == 'body':
            node_colors.append('green')
        else:
            node_colors.append('gray')

    # Create labels using the name attribute
    labels = {node_id: G.nodes[node_id]['name'] for node_id in G.nodes()}

    nx.draw(G, pos, labels=labels, node_color=node_colors,
            node_size=500, font_size=8, font_weight='bold')
    plt.savefig("mujoco_graph.png")
    print("Graph visualization saved to mujoco_graph.png")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=int(1e7),
        help='Total number of time-steps to train model for.'
    )
    hparam = parser.parse_args()

    main(hparam)
