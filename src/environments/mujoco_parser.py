import os
from bs4 import BeautifulSoup as bs
from src.utils.logger_config import logger
from gymnasium.envs.registration import register
import num2words
import numpy as np


TASK_DICT = {
    "Centipede": [3, 5, 7] + [4, 6, 8, 10, 12, 14, 18] + [20, 24, 30, 40, 50],
    "CentipedeTT": [6],
    "CpCentipede": [3, 5, 7] + [4, 6, 8, 10, 12, 14],
    "Reacher": [0, 1, 2, 3, 4, 5, 6, 7],
    "Snake": list(range(3, 10)) + [10, 20, 40],
}

MAX_EPISODE_STEPS_DICT = {
    "Centipede": 1000,
    "CentipedeTT": 1000,
    "CpCentipede": 1000,
    "Snake": 1000,
    "Reacher": 50,
}
REWARD_THRESHOLD = {
    "Centipede": 6000.0,
    "CentipedeTT": 6000.0,
    "CpCentipede": 6000.0,
    "Snake": 360.0,
    "Reacher": -3.75,
}

# walker list
MULTI_TASK_DICT = {
    "MultiWalkers-v1": [
        "WalkersHopper-v1",
        "WalkersHalfhumanoid-v1",
        "WalkersHalfcheetah-v1",
        "WalkersFullcheetah-v1",
        "WalkersOstrich-v1",
    ],
    # just for implementation, only one agent will be run
    "MultiWalkers2Kangaroo-v1": [
        "WalkersHopper-v1",
        "WalkersHalfhumanoid-v1",
        "WalkersHalfcheetah-v1",
        "WalkersFullcheetah-v1",
        "WalkersKangaroo-v1",
    ],
}

# test the robustness of agents
NUM_ROBUSTNESS_AGENTS = 5
ROBUSTNESS_TASK_DICT = {}
for i_agent in range(NUM_ROBUSTNESS_AGENTS + 1):
    ROBUSTNESS_TASK_DICT.update(
        {
            "MultiWalkers"
            + num2words.num2words(i_agent)
            + "-v1": [
                "WalkersHopper" + num2words.num2words(i_agent) + "-v1",
                "WalkersHalfhumanoid" + num2words.num2words(i_agent) + "-v1",
                "WalkersHalfcheetah" + num2words.num2words(i_agent) + "-v1",
                "WalkersFullcheetah" + num2words.num2words(i_agent) + "-v1",
                "WalkersOstrich" + num2words.num2words(i_agent) + "-v1",
            ],
        }
    )
MULTI_TASK_DICT.update(ROBUSTNESS_TASK_DICT)

name_list = []  # record all the environments available

# register the transfer tasks
for env_title, env in ROBUSTNESS_TASK_DICT.items():

    for i_env in env:
        file_name = "environments.multitask_env.walkers:"

        # WalkersHopperone-v1, WalkersHopperoneEnv
        entry_point = file_name + i_env.replace("-v1", "Env")

        register(
            id=i_env,
            entry_point=entry_point,
            max_episode_steps=1000,
            reward_threshold=6000,
        )
        # register the environment name in the name list
        name_list.append(i_env)

# container for multitask experiments
register(
    id="CentipedeContainer-v1",
    entry_point="environments.transfer_env.centipede_env:CentipedeContainer",
    max_episode_steps=MAX_EPISODE_STEPS_DICT["Centipede"],
    reward_threshold=REWARD_THRESHOLD["Centipede"],
)
name_list.append("CentipedeContainer-v1")

# register the robustness tasks
for env in TASK_DICT:
    file_name = "environments.transfer_env." + env.lower() + "_env:"

    for i_part in np.sort(TASK_DICT[env]):
        # NOTE, the order in the name_list actually matters (needed during
        # transfer learning)
        registered_name = (
            env
            + num2words.num2words(i_part)[0].upper()
            + num2words.num2words(i_part)[1:]
        )
        registered_name = registered_name.replace(" ", "").replace("-", "")
        entry_point = file_name.replace("tt", "") + registered_name.replace("TT", "") + "Env"

        register(
            id=(registered_name + "-v1"),
            entry_point=entry_point,
            max_episode_steps=MAX_EPISODE_STEPS_DICT[env],
            reward_threshold=REWARD_THRESHOLD[env],
        )
        # register the environment name in the name list
        name_list.append(registered_name + "-v1")

# register AntS-v1
register(
    id="AntS-v1",
    entry_point="environments.transfer_env.antS:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000,
)

# register the walkers for multi-task learning
register(
    id="WalkersHopper-v1",
    entry_point="environments.multitask_env.walkers:WalkersHopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="WalkersHalfhumanoid-v1",
    entry_point="environments.multitask_env.walkers:WalkersHalfhumanoidEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="WalkersHalfcheetah-v1",
    entry_point="environments.multitask_env.walkers:WalkersHalfcheetahEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="WalkersFullcheetah-v1",
    entry_point="environments.multitask_env.walkers:WalkersFullcheetahEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="WalkersOstrich-v1",
    entry_point="environments.multitask_env.walkers:WalkersOstrichEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="WalkersKangaroo-v1",
    entry_point="environments.multitask_env.walkers:WalkersKangarooEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)


def get_mujoco_model_settings():
    """
        @brief:
            @traditional environments:
                1. Humanoid-v1
                2. HumanoidStandup-v1
                3. HalfCheetah-v1
                4. Hopper-v1
                5. Walker2d-v1
                6. AntS-v1

            @transfer-learning environments:
                1. Centipede
                2. Snake
                4. Reacher
    """
    # step 0: settings about the joint
    JOINT_KEY = ["qpos", "qvel", "qfrc_constr", "qfrc_act"]
    BODY_KEY = ["cinert", "cvel", "cfrc"]

    ROOT_OB_SIZE = {
        "qpos": {"free": 7, "hinge": 1, "slide": 1},
        "qvel": {"free": 6, "hinge": 1, "slide": 1},
        "qfrc_act": {"free": 6, "hinge": 1, "slide": 1},
        "qfrc_constr": {"free": 6, "hinge": 1, "slide": 1},
    }

    # step 1: register the settings for traditional environments
    SYMMETRY_MAP = {
        "Humanoid-v2": 2,
        "HumanoidStandup-v1": 2,
        "HalfCheetah-v3": 1,
        "Hopper-v3": 1,
        "Walker2d-v3": 1,
        "AntS-v1": 2,
        "Swimmer-v3": 2,
        "WalkersHopper-v1": 1,
        "WalkersHalfhumanoid-v1": 1,
        "WalkersHalfcheetah-v1": 1,
        "WalkersFullcheetah-v1": 1,
        "WalkersOstrich-v1": 1,
        "WalkersKangaroo-v1": 1,
    }

    XML_DICT = {
        "Humanoid-v2": "humanoid.xml",
        "HumanoidStandup-v1": "humanoid.xml",
        "HalfCheetah-v3": "half_cheetah.xml",
        "Hopper-v3": "hopper.xml",
        "Walker2d-v3": "walker2d.xml",
        "AntS-v1": "ant.xml",
        "Swimmer-v3": "SnakeThree.xml",
        "WalkersHopper-v1": "WalkersHopper.xml",
        "WalkersHalfhumanoid-v1": "WalkersHalfhumanoid.xml",
        "WalkersHalfcheetah-v1": "WalkersHalfcheetah.xml",
        "WalkersFullcheetah-v1": "WalkersFullcheetah.xml",
        "WalkersOstrich-v1": "WalkersOstrich.xml",
        "WalkersKangaroo-v1": "WalkersKangaroo.xml",
    }
    OB_MAP = {
        "Humanoid-v2": ["qpos", "qvel", "cinert", "cvel", "qfrc_act", "cfrc"],
        "HumanoidStandup-v1": ["qpos", "qvel", "cinert", "cvel", "qfrc_act", "cfrc"],
        "HalfCheetah-v3": ["qpos", "qvel"],
        "Hopper-v3": ["qpos", "qvel"],
        "Walker2d-v3": ["qpos", "qvel"],
        "AntS-v1": ["qpos", "qvel", "cfrc"],
        "Swimmer-v3": ["qpos", "qvel"],
        "WalkersHopper-v1": ["qpos", "qvel"],
        "WalkersHalfhumanoid-v1": ["qpos", "qvel"],
        "WalkersHalfcheetah-v1": ["qpos", "qvel"],
        "WalkersFullcheetah-v1": ["qpos", "qvel"],
        "WalkersOstrich-v1": ["qpos", "qvel"],
        "WalkersKangaroo-v1": ["qpos", "qvel"],
    }

    # step 2: register the settings for the tranfer environments
    SYMMETRY_MAP.update(
        {"Centipede": 2, "CentipedeTT": 2, "CpCentipede": 2, "Snake": 2, "Reacher": 0,}
    )

    OB_MAP.update(
        {
            "Centipede": ["qpos", "qvel", "cfrc"],
            "CentipedeTT": ["qpos", "qvel", "cfrc"],
            "CpCentipede": ["qpos", "qvel", "cfrc"],
            "Snake": ["qpos", "qvel"],
            "Reacher": ["qpos", "qvel", "root_add_5"],
        }
    )
    for env in TASK_DICT:
        for i_part in TASK_DICT[env]:
            registered_name = (
                env
                + num2words.num2words(i_part).replace("-", "")[0].upper()
                + num2words.num2words(i_part).replace("-", "")[1:]
                + "-v1"
            )

            SYMMETRY_MAP[registered_name] = SYMMETRY_MAP[env]
            OB_MAP[registered_name] = OB_MAP[env]
            XML_DICT[registered_name] = registered_name.replace("-v1", ".xml")

    # ob map, symmetry map for robustness task
    for key in ROBUSTNESS_TASK_DICT:
        for env in ROBUSTNESS_TASK_DICT[key]:
            OB_MAP.update({env: ["qpos", "qvel"]})
            SYMMETRY_MAP.update({env: 1})
    # xml dict for botustness task
    for i_agent in range(NUM_ROBUSTNESS_AGENTS + 1):
        XML_DICT.update(
            {
                "WalkersHopper"
                + num2words.num2words(i_agent)
                + "-v1": "WalkersHopper.xml",
                "WalkersHalfhumanoid"
                + num2words.num2words(i_agent)
                + "-v1": "WalkersHalfhumanoid.xml",
                "WalkersHalfcheetah"
                + num2words.num2words(i_agent)
                + "-v1": "WalkersHalfcheetah.xml",
                "WalkersFullcheetah"
                + num2words.num2words(i_agent)
                + "-v1": "WalkersFullcheetah.xml",
                "WalkersOstrich"
                + num2words.num2words(i_agent)
                + "-v1": "WalkersOstrich.xml",
            }
        )

    return SYMMETRY_MAP, XML_DICT, OB_MAP, JOINT_KEY, ROOT_OB_SIZE, BODY_KEY


__all__ = ["parse_mujoco_graph"]

XML_ASSERT_DIR = os.path.join("environments/assets")

"""
    Definition of nodes:
    @root:
        The 'root' type is the combination of the top level 'body' node and
        the top level free 'joint' (two nodes combined)
        Also, additional input will be assigned to the root node
        (e.g. the postion of the targer).

        For different tasks, we should have different MLP for each root.

    @geom, @body, @joint:
        The structure defined in the xml files. Ideally, the MLP for input,
        propogation, and output could be shared among different models.
"""
_NODE_TYPE = ["root", "joint", "body"]  # geom is removed
EDGE_TYPE = {
    "self_loop": 0,
    "root-root": 0,  # root-root is loop, also 0
    "joint-joint": 1,
    "geom-geom": 2,
    "body-body": 3,
    "tendon": 4,
    "joint-geom": 5,
    "geom-joint": -5,  # pc-relationship
    "joint-body": 6,
    "body-joint": -6,
    "body-geom": 7,
    "geom-body": -7,
    "root-geom": 8,
    "geom-root": -8,
    "root-joint": 9,
    "joint-root": -9,
    "root-body": 10,
    "body-root": -10,
}

(
    SYMMETRY_MAP,
    XML_DICT,
    OB_MAP,
    JOINT_KEY,
    ROOT_OB_SIZE,
    BODY_KEY,
) = get_mujoco_model_settings()

# development status
ALLOWED_JOINT_TYPE = ["hinge", "free", "slide"]


def parse_mujoco_graph(
    task_name,
    gnn_node_option="nG,yB",
    xml_path=None,
    root_connection_option="nN, Rn, sE",
    gnn_output_option="shared",
    gnn_embedding_option="shared",
    no_sharing_within_layers=0
):
    """
        @brief:
            get the tree of "geom", "body", "joint" built up.

        @return:
            @tree: This function will return a list of dicts. Every dict
                contains the information of node.

                tree[id_of_the_node]['name']: the unique identifier of the node

                tree[id_of_the_node]['neighbour']: is the list for the
                neighbours

                tree[id_of_the_node]['type']: could be 'geom', 'body' or
                'joint'

                tree[id_of_the_node]['info']: debug info from the xml. it
                should not be used during model-free optimization

                tree[id_of_the_node]['is_output_node']: True or False

            @input_dict: input_dict['id_of_the_node'] = [position of the ob]

            @output_list: This correspond to the id of the node where a output
                is available
    """
    task_name = task_name.replace("TT", "")
    if xml_path is None:  # the xml file is in environments/assets/
        xml_path = os.path.join(XML_ASSERT_DIR, XML_DICT[task_name])

    infile = open(xml_path, "r")
    xml_soup = bs(infile.read(), "xml")
    if "nG" in gnn_node_option:
        # no geom node allowed, this order is very important, 'body' must be
        # after the 'joint'
        node_type_allowed = ["root", "joint", "body"]
    else:
        assert "yG" in gnn_node_option
        node_type_allowed = ["root", "joint", "body", "geom"]
        logger.warning("Using Geom is not a good idea!!\n\n\n")

    # step 0: get the basic information of the nodes ready
    tree, node_type_dict = _get_tree_structure(xml_soup, node_type_allowed)

    # step 1: get the neighbours and relation tree
    tree, relation_matrix = _append_tree_relation(
        tree, node_type_allowed, root_connection_option
    )
    # step 2: get the tendon relationship ready
    # NOTE: no tendons in centipede!
    tree, relation_matrix = _append_tendon_relation(tree, relation_matrix, xml_soup)

    # step 3: get the input list ready
    input_dict, ob_size = _get_input_info(tree, task_name)

    # step 4: get the output list ready
    tree, output_list, output_type_dict, action_size = _get_output_info(
        tree, xml_soup, gnn_output_option
    )

    # step 5: get the node parameters
    node_parameters, para_size_dict = _append_node_parameters(
        tree, xml_soup, node_type_allowed, gnn_embedding_option
    )

    debug_info = {"ob_size": ob_size, "action_size": action_size}

    # step 6: prune the body nodes
    if "nB" in gnn_node_option:
        (
            tree,
            relation_matrix,
            node_type_dict,
            input_dict,
            node_parameters,
            para_size_dict,
        ) = _prune_body_nodes(
            tree=tree,
            relation_matrix=relation_matrix,
            node_type_dict=node_type_dict,
            input_dict=input_dict,
            node_parameters=node_parameters,
            para_size_dict=para_size_dict,
            root_connection_option=root_connection_option,
        )

    # step 7: (optional) uni edge type?
    if "uE" in root_connection_option:
        relation_matrix[np.where(relation_matrix != 0)] = 1
    else:
        assert "sE" in root_connection_option

    if no_sharing_within_layers:
        rows, cols = relation_matrix.shape
        i = 1
        for x in range(0, rows):
            for y in range(0, cols):
                if relation_matrix[x][y] != 0:
                    relation_matrix[x][y] = i
                    i += 1
        new_node_type_dict = {'root': node_type_dict["root"]}
        new_node_parameters = {'root': node_parameters["root"]}
        new_para_size_dict = {'root': 1}
        for typ, par in zip(node_type_dict['joint'], node_parameters['joint']):
            new_node_type_dict[f'joint_{typ}'] = [typ]
            new_node_parameters[f'joint_{typ}'] = [par]
            new_para_size_dict[f'joint_{typ}'] = 1
        node_type_dict, node_parameters, para_size_dict = new_node_type_dict, new_node_parameters, new_para_size_dict

    return dict(
        tree=tree,
        relation_matrix=relation_matrix,
        node_type_dict=node_type_dict,
        output_type_dict=output_type_dict,
        input_dict=input_dict,
        output_list=output_list,
        debug_info=debug_info,
        node_parameters=node_parameters,
        para_size_dict=para_size_dict,
        num_nodes=len(tree),
    )


def _prune_body_nodes(
    tree,
    relation_matrix,
    node_type_dict,
    input_dict,
    node_parameters,
    para_size_dict,
    root_connection_option,
):
    """
        @brief:
            In this function, we will have to remove the body node.
            1. delete all the the bodys, whose ob will be placed into its kid
                joint (multiple joints possible)
            2. for the root node, if kid joint exists, transfer the ownership
                body ob into the kids
    """
    # make sure the tree is structured in a 'root', 'joint', 'body' order
    assert (
        node_type_dict["root"] == [0]
        and max(node_type_dict["joint"]) < min(node_type_dict["body"])
        and "geom" not in node_type_dict
    )

    # for each joint, let it eat its father body root, the relation_matrix and
    # input_dict need to be inherited
    for node_id, i_node in enumerate(tree[0 : min(node_type_dict["body"])]):
        if i_node["type"] != "joint":
            assert i_node["type"] == "root"
            continue

        # find all the parent
        parent = i_node["parent"]

        # inherit the input observation
        if parent in input_dict:
            input_dict[node_id] += input_dict[parent]
        """
            1. inherit the joint with shared body, only the first joint will
                inherit the AB_body's relationship. other joints will be
                attached to the first joint
                A_joint ---- AB_body ---- B_joint. On
            2. inherit joint-joint relationships for sybling joints:
                A_joint ---- A_body ---- B_body ---- B_joint
            3. inherit the root-joint connection
                A_joint ---- A_body ---- root
        """
        # step 1: check if there is brothers / sisters of this node
        children = np.where(relation_matrix[parent, :] == EDGE_TYPE["body-joint"])[0]
        first_brother = [
            child_id
            for child_id in children
            if child_id != node_id and child_id < node_id
        ]
        if len(first_brother) > 0:
            first_brother = min(first_brother)
            relation_matrix[node_id, first_brother] = EDGE_TYPE["joint-joint"]
            relation_matrix[first_brother, node_id] = EDGE_TYPE["joint-joint"]
            continue

        # step 2: the type 2 relationship, note that only the first brother
        # will be considered
        uncles = np.where(relation_matrix[parent, :] == EDGE_TYPE["body-body"])[0]
        for i_uncle in uncles:
            syblings = np.where(relation_matrix[i_uncle, :] == EDGE_TYPE["body-joint"])[
                0
            ]
            if len(syblings) > 0:
                sybling = syblings[0]
            else:
                continue
            if tree[sybling]["parent"] is tree[i_uncle]["parent"]:
                continue
            relation_matrix[node_id, sybling] = EDGE_TYPE["joint-joint"]
            relation_matrix[sybling, node_id] = EDGE_TYPE["joint-joint"]

        # step 3: the type 3 relationship
        uncles = np.where(relation_matrix[parent, :] == EDGE_TYPE["body-root"])[0]
        assert len(uncles) <= 1
        for i_uncle in uncles:
            relation_matrix[node_id, i_uncle] = EDGE_TYPE["joint-root"]
            relation_matrix[i_uncle, node_id] = EDGE_TYPE["root-joint"]

    # remove all the body root
    first_body_node = min(node_type_dict["body"])
    tree = tree[:first_body_node]
    relation_matrix = relation_matrix[:first_body_node, :first_body_node]
    for i_body_node in node_type_dict["body"]:
        if i_body_node in input_dict:
            input_dict.pop(i_body_node)
    node_parameters.pop("body")
    node_type_dict.pop("body")
    para_size_dict.pop("body")

    for i_node in node_type_dict["joint"]:
        assert len(input_dict[i_node]) == len(input_dict[1])

    return (
        tree,
        relation_matrix,
        node_type_dict,
        input_dict,
        node_parameters,
        para_size_dict,
    )


def _get_tree_structure(xml_soup, node_type_allowed):
    mj_soup = xml_soup.find("worldbody").find("body")
    tree = []  # NOTE: the order in the list matters!
    tree_id = 0
    nodes = dict()

    # step 0: set the root node
    motor_names = _get_motor_names(xml_soup)
    node_info = {
        "type": "root",
        "is_output_node": False,
        "name": "root_mujocoroot",
        "neighbour": [],
        "id": 0,
        "info": mj_soup.attrs,
        "raw": mj_soup,
    }
    node_info["attached_joint_name"] = [
        i_joint["name"]
        for i_joint in mj_soup.find_all("joint", recursive=False)
        if i_joint["name"] not in motor_names
    ]
    for key in JOINT_KEY:
        node_info[key + "_size"] = 0
    node_info["attached_joint_info"] = []
    node_info["tendon_nodes"] = []
    tree.append(node_info)
    tree_id += 1

    # step 1: set the 'node_type_allowed' nodes
    for i_type in node_type_allowed:
        nodes[i_type] = mj_soup.find_all(i_type)
        if i_type == "body":
            # the root body should be added to the body
            nodes[i_type] = [mj_soup] + nodes[i_type]
        if len(nodes[i_type]) == 0:
            continue
        for i_node in nodes[i_type]:
            node_info = dict()
            node_info["type"] = i_type
            node_info["is_output_node"] = False
            node_info["raw_name"] = i_node["name"]
            node_info["name"] = node_info["type"] + "_" + i_node["name"]
            node_info["tendon_nodes"] = []
            # this id is the same as the order in the tree
            node_info["id"] = tree_id
            node_info["parent"] = None

            # additional debug information, should not be used during training
            node_info["info"] = i_node.attrs
            node_info["raw"] = i_node

            # NOTE: the get the information about the joint that is directly
            # attached to 'root' node. These joints will be merged into the
            # 'root' node
            if i_type == "joint" and i_node["name"] in tree[0]["attached_joint_name"]:
                tree[0]["attached_joint_info"].append(node_info)
                for key in JOINT_KEY:
                    tree[0][key + "_size"] += ROOT_OB_SIZE[key][i_node["type"]]
                continue

            # currently, only 'hinge' type is supported
            if i_type == "joint" and i_node["type"] not in ALLOWED_JOINT_TYPE:
                logger.warning("NOT IMPLEMENTED JOINT TYPE: {}".format(i_node["type"]))

            tree.append(node_info)
            tree_id += 1

        logger.info("{} {} found".format(len(nodes[i_type]), i_type))
    node_type_dict = {}

    # step 2: get the node_type dict ready
    for i_key in node_type_allowed:
        node_type_dict[i_key] = [
            i_node["id"] for i_node in tree if i_key == i_node["type"]
        ]
        assert len(node_type_dict) >= 1, logger.error(
            "Missing node type {}".format(i_key)
        )
    return tree, node_type_dict


def _append_tree_relation(tree, node_type_allowed, root_connection_option):
    """
        @brief:
            build the relationship matrix and append relationship attribute
            to the nodes of the tree

        @input:
            @root_connection_option:
                'nN, Rn': without neighbour, no additional connection
                'nN, Rb': without neighbour, root connected to all body
                'nN, Ra': without neighbour, root connected to all node
                'yN, Rn': with neighbour, no additional connection
    """
    num_node = len(tree)
    relation_matrix = np.zeros([num_node, num_node], dtype=int)

    # step 1: set graph connection relationship
    for i_node in tree:
        # step 1.1: get the id of the children
        children = i_node["raw"].find_all(recursive=False)
        if len(children) == 0:
            continue
        children_names = [
            i_children.name + "_" + i_children["name"]
            for i_children in children
            if i_children.name in node_type_allowed
        ]
        children_id_list = [
            [node["id"] for node in tree if node["name"] == i_children_name]
            for i_children_name in children_names
        ]

        i_node["children_id_list"] = children_id_list = sum(
            children_id_list, []
        )  # squeeze the list
        current_id = i_node["id"]
        current_type = tree[current_id]["type"]

        # step 1.2: set the children-parent relationship edges
        for i_children_id in i_node["children_id_list"]:
            relation_matrix[current_id, i_children_id] = EDGE_TYPE[
                current_type + "-" + tree[i_children_id]["type"]
            ]
            relation_matrix[i_children_id, current_id] = EDGE_TYPE[
                tree[i_children_id]["type"] + "-" + current_type
            ]
            if tree[current_id]["type"] == "body":
                tree[i_children_id]["parent"] = current_id

        # step 1.3 (optional): set children connected if needed
        if "yN" in root_connection_option:
            for i_node_in_use_1 in i_node["children_id_list"]:
                for i_node_in_use_2 in i_node["children_id_list"]:
                    relation_matrix[i_node_in_use_1, i_node_in_use_2] = EDGE_TYPE[
                        tree[i_node_in_use_1]["type"]
                        + "-"
                        + tree[i_node_in_use_2]["type"]
                    ]

        else:
            assert "nN" in root_connection_option, logger.error(
                "Unrecognized root_connection_option: {}".format(root_connection_option)
            )

    # step 2: set root connection
    if "Ra" in root_connection_option:
        # if root is connected to all the nodes
        for i_node_in_use_1 in range(len(tree)):
            target_node_type = tree[i_node_in_use_1]["type"]

            # add connections between all nodes and root
            relation_matrix[0, i_node_in_use_1] = EDGE_TYPE[
                "root" + "-" + target_node_type
            ]
            relation_matrix[i_node_in_use_1, 0] = EDGE_TYPE[
                target_node_type + "-" + "root"
            ]

    elif "Rb" in root_connection_option:
        for i_node_in_use_1 in range(len(tree)):
            target_node_type = tree[i_node_in_use_1]["type"]

            if not target_node_type == "body":
                continue

            # add connections between body and root
            relation_matrix[0, i_node_in_use_1] = EDGE_TYPE[
                "root" + "-" + target_node_type
            ]
            relation_matrix[i_node_in_use_1, 0] = EDGE_TYPE[
                target_node_type + "-" + "root"
            ]
    else:
        assert "Rn" in root_connection_option, logger.error(
            "Unrecognized root_connection_option: {}".format(root_connection_option)
        )

    # step 3: unset the diagonal terms back to 'self-loop'
    np.fill_diagonal(relation_matrix, EDGE_TYPE["self_loop"])

    return tree, relation_matrix


def _append_tendon_relation(tree, relation_matrix, xml_soup):
    """
        @brief:
            build the relationship of tendon (the spring)
    """
    tendon = xml_soup.find("tendon")
    if tendon is None:
        return tree, relation_matrix
    tendon_list = tendon.find_all("fixed")

    for i_tendon in tendon_list:
        # find the id
        joint_name = ["joint_" + joint["joint"] for joint in i_tendon.find_all("joint")]
        joint_id = [node["id"] for node in tree if node["name"] in joint_name]
        assert len(joint_id) == 2, logger.error(
            "Unsupported tendon: {}".format(i_tendon)
        )

        # update the tree and the relationship matrix
        relation_matrix[joint_id[0], joint_id[1]] = EDGE_TYPE["tendon"]
        relation_matrix[joint_id[1], joint_id[0]] = EDGE_TYPE["tendon"]
        tree[joint_id[0]]["tendon_nodes"].append(joint_id[1])
        tree[joint_id[1]]["tendon_nodes"].append(joint_id[0])

        logger.info(
            "new tendon found between: {} and {}".format(
                tree[joint_id[0]]["name"], tree[joint_id[1]]["name"]
            )
        )

    return tree, relation_matrix


def _get_input_info(tree, task_name):
    input_dict = {}

    joint_id = [node["id"] for node in tree if node["type"] == "joint"]
    body_id = [node["id"] for node in tree if node["type"] == "body"]
    root_id = [node["id"] for node in tree if node["type"] == "root"][0]

    # init the input dict
    input_dict[root_id] = []
    if (
        "cinert" in OB_MAP[task_name]
        or "cvel" in OB_MAP[task_name]
        or "cfrc" in OB_MAP[task_name]
    ):
        candidate_id = joint_id + body_id
    else:
        candidate_id = joint_id
    for i_id in candidate_id:
        input_dict[i_id] = []

    logger.info("scanning ob information...")
    current_ob_id = 0
    for ob_type in OB_MAP[task_name]:

        if ob_type in JOINT_KEY:
            # step 1: collect the root ob's information. Some ob might be
            # ignore, which is specify in the SYMMETRY_MAP
            ob_step = (
                tree[0][ob_type + "_size"]
                - (ob_type == "qpos") * SYMMETRY_MAP[task_name]
            )
            input_dict[root_id].extend(
                list(range(current_ob_id, current_ob_id + ob_step))
            )
            current_ob_id += ob_step

            # step 2: collect the joint ob's information
            for i_id in joint_id:
                input_dict[i_id].append(current_ob_id)
                current_ob_id += 1

        elif ob_type in BODY_KEY:

            BODY_OB_SIZE = 10 if ob_type == "cinert" else 6

            # step 0: skip the 'world' body
            current_ob_id += BODY_OB_SIZE

            # step 1: collect the root ob's information, note that the body will
            # still take this ob
            input_dict[root_id].extend(
                list(range(current_ob_id, current_ob_id + BODY_OB_SIZE))
            )
            # current_ob_id += BODY_OB_SIZE

            # step 2: collect the body ob's information
            for i_id in body_id:
                input_dict[i_id].extend(
                    list(range(current_ob_id, current_ob_id + BODY_OB_SIZE))
                )
                current_ob_id += BODY_OB_SIZE
        else:
            assert "add" in ob_type, logger.error(
                "TYPE {BODY_KEY} NOT RECGNIZED".format(ob_type)
            )
            addition_ob_size = int(ob_type.split("_")[-1])
            input_dict[root_id].extend(
                list(range(current_ob_id, current_ob_id + addition_ob_size))
            )
            current_ob_id += addition_ob_size
        logger.info(
            "after {}, the ob size is reaching {}".format(ob_type, current_ob_id)
        )
    return input_dict, current_ob_id  # to debug if the ob size is matched


def _get_output_info(tree, xml_soup, gnn_output_option):
    output_list = []
    output_type_dict = {}
    motors = xml_soup.find("actuator").find_all("motor")

    for i_motor in motors:
        joint_id = [
            i_node["id"]
            for i_node in tree
            if "joint_" + i_motor["joint"] == i_node["name"]
        ]
        if len(joint_id) == 0:
            # joint_id = 0  # it must be the root if not found
            assert False, logger.error("Motor {} not found!".format(i_motor["joint"]))
        else:
            joint_id = joint_id[0]
        tree[joint_id]["is_output_node"] = True
        output_list.append(joint_id)

        # construct the output_type_dict
        if gnn_output_option == "shared":
            motor_type_name = i_motor["joint"].split("_")[0]
            if motor_type_name in output_type_dict:
                output_type_dict[motor_type_name].append(joint_id)
            else:
                output_type_dict[motor_type_name] = [joint_id]
        elif gnn_output_option == "separate":
            motor_type_name = i_motor["joint"]
            output_type_dict[motor_type_name] = [joint_id]
        else:
            assert gnn_output_option == "unified", logger.error(
                "Invalid output type: {}".format(gnn_output_option)
            )
            if "unified" in output_type_dict:
                output_type_dict["unified"].append(joint_id)
            else:
                output_type_dict["unified"] = [joint_id]

    return tree, output_list, output_type_dict, len(motors)


def _get_motor_names(xml_soup):
    motors = xml_soup.find("actuator").find_all("motor")
    name_list = [i_motor["joint"] for i_motor in motors]
    return name_list


GEOM_TYPE_ENCODE = {
    "capsule": [0.0, 1.0],
    "sphere": [1.0, 0.0],
}


def _append_node_parameters(tree, xml_soup, node_type_allowed, gnn_embedding_option):
    """
        @brief:
            the output of this function is a dictionary.
        @output:
            e.g.: node_parameters['geom'] is a numpy array, which has the shape
            of (num_nodes, para_size_of_'geom')
            the node is ordered in the relative position in the tree
    """
    assert node_type_allowed.index("joint") < node_type_allowed.index("body")

    if gnn_embedding_option == "parameter":
        # step 0: get the para list and default setting for this mujoco xml
        PARAMETERS_LIST, default_dict = _get_para_list(xml_soup, node_type_allowed)

        # step 2: get the node_parameter_list ready, they are in the node_order
        node_parameters = {node_type: [] for node_type in node_type_allowed}
        for node_id in range(len(tree)):
            output_parameter = []

            for i_parameter_type in PARAMETERS_LIST[tree[node_id]["type"]]:
                # collect the information one by one
                output_parameter = _collect_parameter_info(
                    output_parameter,
                    i_parameter_type,
                    tree[node_id]["type"],
                    default_dict,
                    tree[node_id]["info"],
                )

            # this node is finished
            node_parameters[tree[node_id]["type"]].append(output_parameter)

        # step 3: numpy the elements, and do validation check
        for node_type in node_type_allowed:
            node_parameters[node_type] = np.array(
                node_parameters[node_type], dtype=np.float32
            )

        # step 4: get the size of parameters logged
        para_size_dict = {
            node_type: len(node_parameters[node_type][0])
            for node_type in node_type_allowed
        }

        # step 5: trick, root para is going to receive a dummy para [1]
        para_size_dict["root"] = 1
        node_parameters["root"] = np.ones([1, 1])
    elif gnn_embedding_option in ["shared", "noninput_separate", "noninput_shared"]:
        # step 1: preprocess, register the node, get the number of bits for
        # encoding needed
        struct_name_list = {node_type: [] for node_type in node_type_allowed}
        for node_id in range(len(tree)):
            name = tree[node_id]["name"].split("_")
            type_name = name[0]

            if gnn_embedding_option in ["noninput_separate"]:
                register_name = name
                struct_name_list[type_name].append(register_name)
            else:  # shared
                register_name = type_name + "_" + name[1]
                if register_name not in struct_name_list[type_name]:
                    struct_name_list[type_name].append(register_name)
            tree[node_id]["register_embedding_name"] = register_name

        struct_name_list["root"] = [tree[0]["name"]]  # the root
        tree[0]["register_embedding_name"] = tree[0]["name"]

        # step 2: estimate the encoding length
        num_type_bits = 2
        para_size_dict = {  # 2 bits for type encoding
            i_node_type: num_type_bits + 8 for i_node_type in node_type_allowed
        }

        # step 3: get the parameters
        node_parameters = {i_node_type: [] for i_node_type in node_type_allowed}
        appear_str = []
        for node_id in range(len(tree)):
            type_name = tree[node_id]["type"]
            type_str = str(bin(node_type_allowed.index(type_name)))
            type_str = (type_str[2:]).zfill(num_type_bits)
            node_str = str(
                bin(
                    struct_name_list[type_name].index(
                        tree[node_id]["register_embedding_name"]
                    )
                )
            )
            node_str = (node_str[2:]).zfill(para_size_dict[tree[node_id]["type"]] - 2)

            if node_id == 0 or para_size_dict[type_name] == 2:
                node_str = ""

            final_str = type_str + node_str
            if final_str not in appear_str:
                appear_str.append(final_str)

            if "noninput_shared_multi" in gnn_embedding_option:
                node_parameters[type_name].append(
                    tree[node_id]["register_embedding_name"]
                )
            elif "noninput" in gnn_embedding_option:
                node_parameters[type_name].append([appear_str.index(final_str)])
            else:
                node_parameters[type_name].append([int(i_char) for i_char in final_str])

        # step 4: numpy the elements, and do validation check
        if gnn_embedding_option != "noninput_shared_multi":
            para_dtype = (
                np.float32
                if gnn_embedding_option in ["parameter", "shared"]
                else int
            )
            for node_type in node_type_allowed:
                node_parameters[node_type] = np.array(
                    node_parameters[node_type], dtype=para_dtype
                )
    else:
        assert False, logger.error("Invalid option: {}".format(gnn_embedding_option))

    # step 5: postprocess
    # NOTE: make the length of the parameters the same
    if gnn_embedding_option in ["parameter", "shared"]:
        max_length = max([para_size_dict[node_type] for node_type in node_type_allowed])
        for node_type in node_type_allowed:
            shape = node_parameters[node_type].shape
            new_node_parameters = np.zeros([shape[0], max_length], dtype=int)
            new_node_parameters[:, 0 : shape[1]] = node_parameters[node_type]
            node_parameters[node_type] = new_node_parameters
            para_size_dict[node_type] = max_length
    else:
        para_size_dict = {i_node_type: 1 for i_node_type in node_type_allowed}

    return node_parameters, para_size_dict


def _collect_parameter_info(
    output_parameter, parameter_type, node_type, default_dict, info_dict
):
    # step 1: get the parameter str
    if parameter_type in info_dict:
        # append the default setting into default_dict
        para_str = info_dict[parameter_type]
    elif parameter_type in default_dict[node_type]:
        para_str = default_dict[node_type][parameter_type]
    else:
        assert False, logger.error(
            "no information available for node: {}, para: {}".format(
                node_type, parameter_type
            )
        )

    # step 2: parse the str into the parameter numbers
    if node_type == "geom" and para_str in GEOM_TYPE_ENCODE:
        output_parameter.extend(GEOM_TYPE_ENCODE[para_str])
    else:
        output_parameter.extend([float(element) for element in para_str.split(" ")])

    return output_parameter


PARAMETERS_DEFAULT_DICT = {
    "root": {},
    "body": {"pos": "NON_DEFAULT"},
    "geom": {
        "fromto": "-1 -1 -1 -1 -1 -1",
        "size": "NON_DEFAULT",
        "type": "NON_DEFAULT",
    },
    "joint": {
        "armature": "-1",
        "axis": "NON_DEFAULT",
        "damping": "-1",
        "pos": "NON_DEFAULT",
        "stiffness": "-1",
        "range": "-1 -1",
    },
}


def _get_para_list(xml_soup, node_type_allowed):
    """
        @brief:
            for each type in the node_type_allowed, we find the attributes that
            shows up in the xml
            below is the node parameter info list:

            @root (size 0):
                More often the case, the root node is the domain root, as
                there is the 2d/3d information in it.

            @body (max size: 3):
                @pos: 3

            @geom (max size: 9):
                @fromto: 6
                @size: 1
                @type: 2

            @joint (max size: 11):
                @armature: 1
                @axis: 3
                @damping: 1
                @pos: 3
                @stiffness: 1  # important
                @range: 2
    """
    # step 1: get the available parameter list for each node
    para_list = {node_type: [] for node_type in node_type_allowed}
    mj_soup = xml_soup.find("worldbody").find("body")
    for node_type in node_type_allowed:
        # search the node with type 'node_type'
        node_list = mj_soup.find_all(node_type)  # all the nodes
        for i_node in node_list:
            # deal with each node
            for key in i_node.attrs:
                # deal with each attributes
                if (
                    key not in para_list[node_type]
                    and key in PARAMETERS_DEFAULT_DICT[node_type]
                ):
                    para_list[node_type].append(key)

    # step 2: get default parameter settings
    default_dict = PARAMETERS_DEFAULT_DICT
    default_soup = xml_soup.find("default")
    if default_soup is not None:
        for node_type, para_type_list in para_list.items():
            # find the default str if possible
            type_soup = default_soup.find(node_type)
            if type_soup is not None:
                for para_type in para_type_list:
                    if para_type in type_soup.attrs:
                        default_dict[node_type][para_type] = type_soup[para_type]
            else:
                logger.info(
                    "No default settings available for type {}".format(node_type)
                )
    else:
        logger.warning("No default settings available for this xml!")

    return para_list, default_dict


def parse_mujoco_template(
    task_name,
    input_feat_dim,
    gnn_node_option,
    root_connection_option,
    gnn_output_option,
    gnn_embedding_option,
    no_sharing_within_layers
):
    """
             @brief:
                 In this function, we construct the dict for node information.
                 The structure is _node_info
             @attribute:
                 1. general informatin about the graph
                     @self._node_info['tree']
                     @self._node_info['debug_info']
                     @self._node_info['relation_matrix']

                 2. information about input output
                     @self._node_info['input_dict']:
                         self._node_info['input_dict'][id_of_node] is a list of
                         ob positions
                     @self._node_info['output_list']

                 3. information about the node
                     @self._node_info['node_type_dict']:
                         self._node_info['node_type_dict']['body'] is a list of
                         node id
                     @self._node_info['num_nodes']

                 4. information about the edge
                     @self._node_info['edge_type_list'] = self._edge_type_list
                         the list of edge ids
                     @self._node_info['num_edges']
                     @self._node_info['num_edge_type']

                 6. information about the index
                     @self._node_info['node_in_graph_list']
                         The order of nodes if placed by types ('joint', 'body')
                     @self._node_info['inverse_node_list']
                         The inverse of 'node_in_graph_list'
                     @self._node_info['receive_idx'] = receive_idx
                     @self._node_info['receive_idx_raw'] = receive_idx_raw
                     @self._node_info['send_idx'] = send_idx

                 7. information about the embedding size and ob size
                     @self._node_info['para_size_dict']
                     @self._node_info['ob_size_dict']
             """
    # step 0: parse the mujoco xml
    node_info = parse_mujoco_graph(
        task_name,
        gnn_node_option=gnn_node_option,
        root_connection_option=root_connection_option,
        gnn_output_option=gnn_output_option,
        gnn_embedding_option=gnn_embedding_option,
        no_sharing_within_layers=no_sharing_within_layers,
    )

    # step 2: check for ob size for each node type, construct the node dict
    node_info = construct_ob_size_dict(node_info, input_feat_dim)

    # step 3: get the inverse node offsets (used to construct gather idx)
    node_info = get_inverse_type_offset(node_info, "node")

    # step 4: get the inverse node offsets (used to gather output idx)
    node_info = get_inverse_type_offset(node_info, "output")

    # step 5: register existing edge and get the receive index and send index
    node_info = get_receive_send_idx(node_info)

    return node_info


def construct_ob_size_dict(node_info, input_feat_dim):
    """
        @brief: for each node type, we collect the ob size for this type
    """
    node_info["ob_size_dict"] = {}
    for node_type in node_info["node_type_dict"]:
        node_ids = node_info["node_type_dict"][node_type]

        # record the ob_size for each type of node
        if node_ids[0] in node_info["input_dict"]:
            node_info["ob_size_dict"][node_type] = len(
                node_info["input_dict"][node_ids[0]]
            )
        else:
            node_info["ob_size_dict"][node_type] = 0

        node_ob_size = [
            len(node_info["input_dict"][node_id])
            for node_id in node_ids
            if node_id in node_info["input_dict"]
        ]

        if len(node_ob_size) == 0:
            continue

        assert node_ob_size.count(node_ob_size[0]) == len(node_ob_size), logger.error(
            "Nodes (type {}) have wrong ob size: {}!".format(node_type, node_ob_size)
        )

    return node_info


def get_inverse_type_offset(node_info, mode):
    assert mode in ["output", "node"], logger.error("Invalid mode: {}".format(mode))
    node_info["inverse_" + mode + "_extype_offset"] = []
    node_info["inverse_" + mode + "_intype_offset"] = []
    node_info["inverse_" + mode + "_self_offset"] = []
    node_info["inverse_" + mode + "_original_id"] = []
    current_offset = 0
    for mode_type in node_info[mode + "_type_dict"]:
        i_length = len(node_info[mode + "_type_dict"][mode_type])
        # the original id
        node_info["inverse_" + mode + "_original_id"].extend(
            node_info[mode + "_type_dict"][mode_type]
        )

        # In one batch, how many element is listed before this type?
        # e.g.: [A, A, C, B, C, A], with order [A, B, C] --> [0, 0, 4, 3, 4, 0]
        node_info["inverse_" + mode + "_extype_offset"].extend(
            [current_offset] * i_length
        )

        # In current type, what is the position of this node?
        # e.g.: [A, A, C, B, C, A] --> [0, 1, 0, 0, 1, 2]
        node_info["inverse_" + mode + "_intype_offset"].extend(list(range(i_length)))

        # how many nodes are in this type?
        # e.g.: [A, A, C, B, C, A] --> [3, 3, 2, 1, 2, 3]
        node_info["inverse_" + mode + "_self_offset"].extend([i_length] * i_length)
        current_offset += i_length

    sorted_id = np.array(node_info["inverse_" + mode + "_original_id"])
    sorted_id.sort()
    node_info["inverse_" + mode + "_original_id"] = [
        node_info["inverse_" + mode + "_original_id"].index(i_node)
        for i_node in sorted_id
    ]

    node_info["inverse_" + mode + "_extype_offset"] = np.array(
        [
            node_info["inverse_" + mode + "_extype_offset"][i_node]
            for i_node in node_info["inverse_" + mode + "_original_id"]
        ]
    )
    node_info["inverse_" + mode + "_intype_offset"] = np.array(
        [
            node_info["inverse_" + mode + "_intype_offset"][i_node]
            for i_node in node_info["inverse_" + mode + "_original_id"]
        ]
    )
    node_info["inverse_" + mode + "_self_offset"] = np.array(
        [
            node_info["inverse_" + mode + "_self_offset"][i_node]
            for i_node in node_info["inverse_" + mode + "_original_id"]
        ]
    )

    return node_info


def get_receive_send_idx(node_info):
    # register the edges that shows up, get the number of edge type
    edge_dict = EDGE_TYPE
    edge_type_list = []  # if one type of edge exist, register

    for edge_id in range(1000):
        if edge_id == 0:
            continue  # the self loop is not considered here
        if (node_info["relation_matrix"] == edge_id).any():
            edge_type_list.append(edge_id)

    node_info["edge_type_list"] = edge_type_list
    node_info["num_edge_type"] = len(edge_type_list)

    receive_idx_raw = {}
    receive_idx = []
    send_idx = {}
    for edge_type in node_info["edge_type_list"]:
        receive_idx_raw[edge_type] = []
        send_idx[edge_type] = []
        i_id = np.where(node_info["relation_matrix"] == edge_type)
        for i_edge in range(len(i_id[0])):
            send_idx[edge_type].append(i_id[0][i_edge])
            receive_idx_raw[edge_type].append(i_id[1][i_edge])
            receive_idx.append(i_id[1][i_edge])

    node_info["receive_idx"] = receive_idx
    node_info["receive_idx_raw"] = receive_idx_raw
    node_info["send_idx"] = send_idx
    node_info["num_edges"] = len(receive_idx)

    return node_info
