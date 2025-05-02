########################################################
### edited from https://github.com/tommasomarzi/fgrl ###
########################################################

import xmltodict
import os
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from shutil import copyfile
import numpy as np
import torch
import networkx as nx
import re


class MujocoParser:
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # Retrieve MuJoCo XML files for training
        envs_train_names = [self.env_name]
        self.morph_graphs = dict()

        for name in envs_train_names:
            if name + '.xml' in os.listdir(self.xml_path):
                xml_file = os.path.join(self.xml_path, '{}.xml'.format(name))
                self.morph_graphs[name] = get_graph_structure(xml_file)
        envs_train_names.sort()

        # sort envs acc to decreasing order of number of limbs (required due to issue in DummyVecEnv)
        order_idx = np.argsort([len(self.morph_graphs[env_name]) for env_name in envs_train_names])[::-1]
        envs_train_names = [envs_train_names[order] for order in order_idx]

        # Set up training env ================================================
        self.limb_obs_size, self.max_action = self.register_envs(envs_train_names, self.max_episodic_timesteps)
        self.max_num_limbs = max([len(self.morph_graphs[env_name]) for env_name in envs_train_names])

        # create vectorized training env
        obs_max_len = max([len(self.morph_graphs[env_name]) for env_name in envs_train_names]) * self.limb_obs_size
        self.envs_train = [self.make_env_wrapper(name, obs_max_len, self.seed) for name in envs_train_names]

        # self.envs_train = DummyVecEnv(envs_train)  # vectorized env (necessary for multiprocessing)

        # determine the maximum number of children in all the training envs
        self.max_children = self.find_max_children(envs_train_names, self.morph_graphs)

        # Get graph node features (distance from torso)
        self.graph_n_feats = dict()
        if self.enable_features:
            node_feats_dim = 1  # adding only one feature (distance from torso)
            for g_name, g_struct in self.morph_graphs.items():
                edge_list = np.array([g_struct[1:], np.arange(1, len(g_struct))]).T.tolist()
                g = nx.DiGraph(edge_list)
                node_feats = (np.array([nx.shortest_path_length(g, 0, n_id) for n_id in range(len(g_struct))]
                                       ).reshape(-1, 1))
                self.graph_n_feats[g_name] = torch.tensor(node_feats, dtype=torch.long).view(-1, node_feats_dim)
        else:
            for env in envs_train_names:
                self.graph_n_feats[env] = None

    def register_envs(self, env_names, max_episode_steps, custom_xml=False):
        """
        register the MuJoCo envs with Gym and return the per-limb observation size and max action value
        (for modular policy training)
        """
        # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
        paths_to_register = []
        # existing envs
        if not custom_xml:
            for name in env_names:
                paths_to_register.append(os.path.join(self.xml_path, "{}.xml".format(name)))
        # custom envs
        else:
            if os.path.isfile(custom_xml):
                paths_to_register.append(custom_xml)
            elif os.path.isdir(custom_xml):
                for name in sorted(os.listdir(custom_xml)):
                    if '.xml' in name:
                        paths_to_register.append(os.path.join(custom_xml, name))
        # register each env
        for xml in paths_to_register:
            env_name = os.path.basename(xml)[:-4]
            env_file = env_name
            # create a copy of modular environment for custom xml model
            if not os.path.exists(os.path.join(self.env_dir, '{}.py'.format(env_name))):
                # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
                copyfile(self.base_modular_env_path, '{}.py'.format(os.path.join(self.env_dir, env_name)))
            params = {'xml': os.path.abspath(xml)}
            # register with gym (check how it works)
            register(id=("%s-v0" % env_name),
                     max_episode_steps=max_episode_steps,
                     entry_point="src.environments.%s:ModularEnv" % env_file,
                     kwargs=params)
            env = IdentityWrapper(gym.make("src.environments:%s-v0" % env_name))
            # the following is the same for each env
            limb_obs_size = env.limb_obs_size
            max_action = env.max_action
        return limb_obs_size, max_action

    @staticmethod
    def find_max_children(env_names, graphs):
        """return the maximum number of children given a list of env names and their corresponding graph structures"""
        max_children = 0
        for name in env_names:
            most_frequent = max(graphs[name], key=graphs[name].count)
            max_children = max(max_children, graphs[name].count(most_frequent))
        return max_children

    @staticmethod
    def make_env_wrapper(env_name, obs_max_len=None, seed=0):
        """return wrapped gym environment for parallel sample collection (vectorized environments)"""
        e = gym.make("src.environments:%s-v0" % env_name, seed=seed, render_mode='human')
        e.reset()
        e = ModularEnvWrapper(e, obs_max_len)
        return e


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
    q: 1x4 quaternion
    Returns
    r: 1x3 exponential map
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    if np.abs(np.linalg.norm(q) - 1) > 1e-3:
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")
    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)
    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0
    r = r0 * theta
    return r


def get_graph_structure(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the graph structure as a parents list"""

    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch, self_idx)

    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    try:
        root = xml['mujoco']['worldbody']['body']
        assert not isinstance(root, list), (
            'worldbody can only contain one body (torso) for the current implementation, but found {}'.format(root))
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    # signal message flipping for flipped walker morphologies
    if 'walker' in os.path.basename(xml_file) and 'flipped' in os.path.basename(xml_file):
        parents[0] = -2
    return parents


def get_graph_joints(xml_file):
    """
    Traverse the given xml file as a tree by pre-order and return all the joints defined as a list of tuples
    (body_name, joint_name1, ...) for each body
    Used to match the order of joints defined in worldbody and joints defined in actuators
    """

    def preorder(b):
        if 'joint' in b:
            if isinstance(b['joint'], list) and b['@name'] != 'torso':
                raise Exception("The given xml file does not follow the standard MuJoCo format.")
            elif not isinstance(b['joint'], list):
                b['joint'] = [b['joint']]
            joints.append([b['@name']])
            for j in b['joint']:
                joints[-1].append(j['@name'])
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch)

    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml['mujoco']['worldbody']['body']
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    return joints


def get_motor_joints(xml_file):
    """
    Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators
    Used to match the order of joints defined in worldbody and joints defined in actuators
    """
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml['mujoco']['actuator']['motor']
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m['@joint'])
    return joints


class IdentityWrapper(gym.Wrapper):
    """wrapper with useful attributes and helper functions"""
    def __init__(self, env):
        super(IdentityWrapper, self).__init__(env)
        self.num_limbs = self.env.unwrapped.model.nbody - 1
        self.limb_obs_size = self.env.unwrapped.observation_space.shape[0] // self.num_limbs
        self.max_action = float(self.env.unwrapped.action_space.high[0])


class ModularEnvWrapper(gym.Wrapper):
    """
    Force env to return fixed shape obs when called .reset() and .step() and removes action's padding before execution
    Also match the order of the actions returned by modular policy to the order of the environment actions
    """
    def __init__(self, env, obs_max_len=None):
        super(ModularEnvWrapper, self).__init__(env)
        # if no max length specified for obs, use the current env's obs size
        if obs_max_len:
            self.obs_max_len = obs_max_len
        else:
            self.obs_max_len = self.env.observation_space.shape[0]
        self.action_len = self.env.action_space.shape[0]
        self.num_limbs = self.env.unwrapped.model.nbody-1
        self.limb_obs_size = self.env.observation_space.shape[0] // self.num_limbs
        self.max_action = float(self.env.action_space.high[0])
        self.xml = self.env.unwrapped.xml
        self.model = env.unwrapped.model

        self.action_mapping = self.initialize_action_mapping()

    def initialize_action_mapping(self):
        mapping = {}
        for i in range(self.model.nu):  # nu is the number of actuators
            actuator_id = i
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            mapping[i] = joint_id - 1
        return mapping

    def step(self, action):
        action = action[:self.num_limbs] # clip the 0-padding before processing
        reordered_action = np.zeros_like(action)
        for logical_idx, env_idx in self.action_mapping.items():
            if logical_idx < len(action):
                reordered_action[env_idx] = action[logical_idx]
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        assert len(obs) <= self.obs_max_len, "env's obs has length {}, which exceeds initiated obs_max_len {}".format(
            len(obs), self.obs_max_len)
        obs = np.append(obs, np.zeros((self.obs_max_len - len(obs))))
        return obs, reward, done, False, info

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)[0]
        assert len(obs) <= self.obs_max_len, "env's obs has length {}, which exceeds initiated obs_max_len {}".format(
            len(obs), self.obs_max_len)
        obs = np.append(obs, np.zeros((self.obs_max_len - len(obs))))
        return obs


def create_edges(env, device):
    parent_list = get_graph_structure(env.unwrapped.xml)
    edges = []
    for i, j in enumerate(parent_list):
        if j != -1:
            edges.append([i, j])
            edges.append([j, i])
    if edges:
        return torch.tensor(edges, dtype=torch.long, device=device).t()
    else:
        return torch.zeros((2, 0), dtype=torch.long, device=device)


def check_actuators(env):
    joint_list = get_graph_joints(env.unwrapped.xml)
    mask = [not re.search(r'torso', sublist[0], re.IGNORECASE) for sublist in joint_list]
    new_list = [joint_list[i][1] for i in range(len(joint_list)) if mask[i]]
    assert new_list == get_motor_joints(env.unwrapped.xml), \
        (f"Actuator ordering in XML file does not match Mujoco's expected ordering. Please search for <actuator> in the"
         f"xml file and rearrange the motor objects to match Mujoco's expected ordering: {new_list}")
    return mask