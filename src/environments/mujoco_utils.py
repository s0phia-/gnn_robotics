########################################################
### edited from https://github.com/tommasomarzi/fgrl ###
########################################################
import xmltodict
import os
from gymnasium.envs.registration import register
import gymnasium as gym
from shutil import copyfile
import numpy as np
from src.main import XML_PATH


class MujocoParser:
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items())

    def stuff(self):
        # Retrieve MuJoCo XML files for training
        envs_train_names = []
        morph_graphs = dict()

        for morphology in self.env_name:
            envs_train_names += [name[:-4] for name in os.listdir(XML_PATH) if '.xml' in name and morphology in name]

        for name in envs_train_names:
            xml_path = os.path.join(XML_PATH, '{}.xml'.format(name))
            morph_graphs[name] = MujocoGraph(xml_path).getGraphStructure

        envs_train_names.sort()

        # sort envs acc to decreasing order of number of limbs (required due to issue in DummyVecEnv)
        order_idx = np.argsort([len(self.morph_graphs[env_name]) for env_name in envs_train_names])[::-1]
        envs_train_names = [envs_train_names[order] for order in order_idx]

        print('Training envs: {}'.format(envs_train_names))
        print('Population size: {}'.format(self.population_size))
        print('Seed: {}\n'.format(self.cfg.seed))

        # Set up training env ================================================
        self.limb_obs_size, self.max_action = register_envs(envs_train_names, self.max_episodic_timesteps)
        self.cfg.limb_obs_size = self.limb_obs_size

        self.cfg.max_num_limbs = self.max_num_limbs = max([len(self.morph_graphs[env_name])
                                                           for env_name in envs_train_names])
        # create vectorized training env
        obs_max_len = max([len(self.morph_graphs[env_name]) for env_name in envs_train_names]) * self.limb_obs_size
        envs_train = [utils.makeEnvWrapper(name, obs_max_len, self.cfg.seed) for name in envs_train_names]

        self.envs_train = DummyVecEnv(envs_train)  # vectorized env (necessary for multiprocessing)
        self.envs = Envs
        self.envs.envs_train = self.envs_train

        # determine the maximum number of children in all the training envs
        self.max_children = utils.findMaxChildren(envs_train_names, self.morph_graphs)

        self.cfg.morph_graphs = self.morph_graphs


class IdentityWrapper(gym.Wrapper):
    """wrapper with useful attributes and helper functions"""

    def __init__(self, env):
        super(IdentityWrapper, self).__init__(env)
        self.num_limbs = len(self.env.model.body_names[1:])
        self.limb_obs_size = self.env.observation_space.shape[0] // self.num_limbs
        self.max_action = float(self.env.action_space.high[0])


class MujocoGraph:
    def __init__(self, xml_file):
        self.xml_file = xml_file

    def get_graph_structure(self):
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

        with open(self.xml_file) as fd:
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
        if 'walker' in os.path.basename(self.xml_file) and 'flipped' in os.path.basename(self.xml_file):
            parents[0] = -2
        return parents

    def get_graph_joints(self):
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

        with open(self.xml_file) as fd:
            xml = xmltodict.parse(fd.read())
        joints = []
        try:
            root = xml['mujoco']['worldbody']['body']
        except:
            raise Exception("The given xml file does not follow the standard MuJoCo format.")
        preorder(root)
        return joints

    def get_motor_joints(self):
        """
        Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators
        Used to match the order of joints defined in worldbody and joints defined in actuators
        """
        with open(self.xml_file) as fd:
            xml = xmltodict.parse(fd.read())
        joints = []
        motors = xml['mujoco']['actuator']['motor']
        if not isinstance(motors, list):
            motors = [motors]
        for m in motors:
            joints.append(m['@joint'])
        return joints


ENV_DIR = './environments'
XML_DIR = './environments/assets'
BASE_MODULAR_ENV_PATH = './environments/ModularEnv.py'
DATA_DIR = './results'


def register_envs(env_names, max_episode_steps, custom_xml):
    """
    register the MuJoCo envs with Gym and return the per-limb observation size and max action value
    (for modular policy training)
    """
    # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
    paths_to_register = []
    # existing envs
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, "{}.xml".format(name)))
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
        if not os.path.exists(os.path.join(ENV_DIR, '{}.py'.format(env_name))):
            # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
            copyfile(BASE_MODULAR_ENV_PATH, '{}.py'.format(os.path.join(ENV_DIR, env_name)))
        params = {'xml': os.path.abspath(xml)}
        # register with gym (check how it works)
        register(id=("%s-v0" % env_name),
                 max_episode_steps=max_episode_steps,
                 entry_point="environments.%s:ModularEnv" % env_file,
                 kwargs=params)
        env = IdentityWrapper(gym.make("environments:%s-v0" % env_name))
        # the following is the same for each env
        limb_obs_size = env.limb_obs_size
        max_action = env.max_action
    return limb_obs_size, max_action


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
