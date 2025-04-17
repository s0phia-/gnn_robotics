import torch
from src.environments.mujoco_parser import get_graph_structure, get_graph_joints, get_motor_joints


def create_edges(env):
    parent_list = get_graph_structure(env.unwrapped.xml)
    edges = []
    for i, j in enumerate(parent_list):
        if j != -1:
            edges.append([i, j])
            edges.append([j, i])
    if edges:
        return torch.tensor(edges, dtype=torch.long).t()
    else:
        return torch.zeros((2, 0), dtype=torch.long)


def create_actuator_mapping(env):
    joint_list = get_graph_joints(env.unwrapped.xml)
    actuator_list = get_motor_joints(env.unwrapped.xml)
    mapping_dict = {}

    for node_idx, (_, joint_name) in enumerate(joint_list):
        if joint_name in actuator_list:
            actuator_idx = actuator_list.index(joint_name)
            mapping_dict[node_idx] = actuator_idx

    def actuator_mapping(node_outputs):
        actuator_actions = torch.zeros(len(actuator_list), device=node_outputs.device)
        for node_idx, actuator_idx in mapping_dict.items():
            actuator_actions[actuator_idx] = node_outputs[node_idx]
        return actuator_actions

    return actuator_mapping
