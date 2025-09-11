########################################################
### edited from https://github.com/tommasomarzi/fgrl ###
########################################################

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from src.environments.mujoco_parser import quat2expmap
from gymnasium.spaces import Box


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        # "render_fps": 25,
    }

    def __init__(self, xml, seed=None, **kwargs):
        print(f"HERE: self.metadata: {self.metadata}")
        self.xml = xml
        self.num_nodes, self.edge_idx, self.mask = None, None, None

        render_mode = kwargs.get('render_mode', None)
        self._desired_render_mode = render_mode
        print(f"{self.xml=}")
        mujoco_env.MujocoEnv.__init__(self, model_path=xml,
                                      frame_skip=4,
                                      observation_space=None,
                                      render_mode=None, )
        utils.EzPickle.__init__(self)

        if seed is not None:
            self.reset(seed=seed)
        else:
            self.reset()
        self.num_limbs = self.model.nbody - 1
        self.limb_obs_size = len(self._get_obs()) // self.num_limbs
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_limbs * self.limb_obs_size,),
                                     dtype=np.float32)

    def step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.data.qpos[0]
        alive_bonus = 1.0
        reward_ctrl = - 0.1 * np.square(a).sum()
        reward_run = (posafter - posbefore) / self.dt
        reward = reward_ctrl + reward_run + alive_bonus
        terminated = False
        truncated = False
        ob = self._get_obs()
        if hasattr(reward, 'item'):
            reward = float(reward.item())
        else:
            reward = float(reward)
        return ob, reward, terminated, truncated, self._get_reset_info()

    def _get_obs(self):
        """
        this function loops through numbers 1...num_joints, gets features, and concatenates them together in that order.
        """

        def _get_obs_per_limb(b):
            if b == 'torso':
                limb_type_vec = np.array((1, 0, 0, 0))
            elif 'thigh' in b:
                limb_type_vec = np.array((0, 1, 0, 0))
            elif 'shin' in b:
                limb_type_vec = np.array((0, 0, 1, 0))
            elif 'foot' in b:
                limb_type_vec = np.array((0, 0, 0, 1))
            else:
                limb_type_vec = np.array((0, 0, 0, 0))

            torso_id = self.data.body("torso").id
            torso_x_pos = self.data.xpos[torso_id][0]
            body_id = self.data.body(b).id
            xpos = self.data.xpos[body_id].copy()
            xpos[0] -= torso_x_pos

            q = self.data.xquat[body_id]
            expmap = quat2expmap(q)

            xvelp = np.clip(self.data.cvel[body_id][:3], -10, 10)  # Linear velocity
            xvelr = self.data.cvel[body_id][3:]  # Angular velocity

            obs = np.concatenate([xpos, xvelp, xvelr, expmap, limb_type_vec])

            # include current joint angle and joint range as input
            if body_id == torso_id:
                angle = 0.
                joint_range = [0., 0.]
            else:
                jnt_adr = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[jnt_adr]  # Assuming each body has only one joint
                angle = np.degrees(self.data.qpos[qpos_adr])  # angle of current joint, scalar
                joint_range = np.degrees(self.model.jnt_range[jnt_adr])  # range of current joint, (2,)
                # normalize
                angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                joint_range[0] = (180. + joint_range[0]) / 360.
                joint_range[1] = (180. + joint_range[1]) / 360.
            obs = np.concatenate([obs, [angle], joint_range])
            return obs

        full_obs = np.concatenate([_get_obs_per_limb(i) for i in ['bfoot', 'bshin', 'bthigh', 'ffoot', 'fshin',
                                                                  'fthigh', 'torso']])
        return full_obs.ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def render(self):
        if hasattr(self, 'mujoco_renderer'):
            # Enable the stored render mode when explicitly called
            if self._desired_render_mode is not None and self.render_mode != self._desired_render_mode:
                self.render_mode = self._desired_render_mode
                self.mujoco_renderer.render_mode = self._desired_render_mode

            # Make camera follow the agent
            if self.mujoco_renderer.viewer is not None:
                # Get torso position
                torso_id = self.data.body("torso").id
                torso_pos = self.data.xpos[torso_id]

                # Set camera to follow torso
                self.mujoco_renderer.viewer.cam.lookat[0] = torso_pos[0]
                self.mujoco_renderer.viewer.cam.lookat[1] = torso_pos[1]
                self.mujoco_renderer.viewer.cam.lookat[2] = torso_pos[2]

        return super().render()
