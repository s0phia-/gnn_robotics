import os
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


def _to_np(x):
    return np.array(x)


# --- CHANGE 1: Add 'node_labels' as an argument to the function ---
def plot_attention(edge_index, attention_scores, node_labels=None, save_path=None):
    edge = _to_np(edge_index)
    scores = _to_np(attention_scores)

    edge = np.asarray(edge)
    if edge.ndim != 2 or (edge.shape[0] != 2 and edge.shape[1] != 2):
        raise ValueError(f'edge_index must be shape (2, E) or (E, 2); got {edge.shape}')
    edges_rows = edge.T if edge.shape[0] == 2 else edge
    edge_list = [tuple(map(int, e)) for e in edges_rows]

    # collapse multi-dim scores to one value per edge
    scores = np.asarray(scores)
    # if there are no edges, nothing to plot
    if len(edge_list) == 0:
        print('No edges to plot; exiting.')
        return
    if scores.ndim > 1:
        if scores.shape[0] == len(edge_list):
            scores = scores.mean(axis=1)
        elif scores.shape[-1] == len(edge_list):
            scores = scores.mean(axis=0)
        else:
            # reshape to (E, -1) and average
            scores = scores.reshape(len(edge_list), -1).mean(axis=1)

    scores = np.asarray(scores, dtype=float).ravel()
    scores = np.nan_to_num(scores, nan=0.0)

    if scores.size == 1 and len(edge_list) > 1:
        scores = np.full(len(edge_list), scores.item())
    if scores.size != len(edge_list):
        raise ValueError(f'Number of edges ({len(edge_list)}) != number of attention values ({scores.size})')

    # Merge bidirectional edges: average scores for undirected pair keys
    undirected_map = {}
    for (u, v), s in zip(edge_list, scores.tolist()):
        key = tuple(sorted((int(u), int(v))))
        if key in undirected_map:
            undirected_map[key].append(s)
        else:
            undirected_map[key] = [s]
    merged_edges = []
    merged_scores = []
    for (u, v), vals in undirected_map.items():
        merged_edges.append((u, v))
        merged_scores.append(float(np.mean(vals)))

    merged_scores = np.array(merged_scores, dtype=float)

    # Increase visual contrast: normalize to [0,1], then apply gamma < 1 (stretch)
    vmin = merged_scores.min()
    vmax = merged_scores.max()
    if vmax - vmin < 1e-12:
        norm_scores = np.zeros_like(merged_scores)
    else:
        norm_scores = (merged_scores - vmin) / (vmax - vmin)
    gamma = 0.6
    contrast_scores = np.power(norm_scores, gamma)

    G = nx.Graph()
    G.add_edges_from(merged_edges)
    n = int(edge.max()) + 1
    G.add_nodes_from(range(n))
    
    # --- AMENDED LAYOUT LOGIC ---
    # Default to spring layout
    pos = nx.spring_layout(G)
    # If node labels are provided, attempt to create a fixed layout
    if node_labels is not None:
        labels_dict = dict(node_labels)
        # Define the coordinates for each specific node label
        coord_map = {
            'torso': (0, 0), 'aux_1': (1, 1), 'f_1': (2, 2),
            'aux_2': (1, -1), 'f_2': (2, -2), 'aux_3': (-1, 1),
            'f_3': (-2, 2), 'aux_4': (-1, -1), 'f_4': (-2, -2),
        }
        
        fixed_pos = {}
        for node_index, node_name in labels_dict.items():
            if node_name in coord_map:
                fixed_pos[node_index] = coord_map[node_name]

        # Use the fixed layout only if all nodes could be mapped
        if len(fixed_pos) == len(G.nodes()):
            pos = fixed_pos
        else:
            print("Warning: Not all nodes could be mapped to fixed positions. Using spring layout as fallback.")


    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=120)
    nx.draw_networkx_edges(
        G, pos, edgelist=merged_edges,
        edge_color=contrast_scores, edge_cmap=plt.cm.viridis, width=2
    )
    
    # --- CHANGE 2: Convert input to a dictionary and draw the labels ---
    if node_labels is not None:
        # The dict() constructor handles enumerate objects and lists of tuples perfectly
        labels_dict = dict(node_labels)
        nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, font_color='black')


    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()
        