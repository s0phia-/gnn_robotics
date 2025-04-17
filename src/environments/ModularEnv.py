########################################################
### edited from https://github.com/tommasomarzi/fgrl ###
########################################################

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from src.environments.mujoco_parser import quat2expmap
from gymnasium.spaces import Box


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25,
    }
    def __init__(self, xml, seed=None, **kwargs):
        self.xml = xml
        render_mode = kwargs.get('render_mode', None)
        mujoco_env.MujocoEnv.__init__(self, model_path=xml,
                                      frame_skip=4,
                                      observation_space=Box(low=-np.inf, high=np.inf, shape=(135,), dtype=float),
                                      render_mode=render_mode,)
        utils.EzPickle.__init__(self)
        if seed is not None:
            self.reset(seed=seed)
        else:
            self.reset()

    def step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.data.qpos[0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = False
        ob = self._get_obs()
        return ob, reward, done, False, {}

    def _get_obs(self):
        def _get_obs_per_limb(body_id):
            # Get the torso position
            torso_id = self.data.body("torso").id
            torso_x_pos = self.data.xpos[torso_id][0]

            # Get body position
            xpos = self.data.xpos[body_id].copy()
            xpos[0] -= torso_x_pos

            # Get quaternion and convert to expmap
            q = self.data.xquat[body_id]
            expmap = quat2expmap(q)

            # Get velocities
            xvelp = np.clip(self.data.cvel[body_id][:3], -10, 10)  # Linear velocity
            xvelr = self.data.cvel[body_id][3:]  # Angular velocity

            obs = np.concatenate([xpos, xvelp, xvelr, expmap])

            # Include current joint angle and joint range as input
            if body_id == torso_id:
                angle = 0.
                joint_range = [0., 0.]
            else:
                jnt_adr = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[jnt_adr]  # Assuming each body has only one joint
                angle = np.degrees(self.data.qpos[qpos_adr])  # Angle of current joint, scalar
                joint_range = np.degrees(self.model.jnt_range[jnt_adr])  # Range of current joint, (2,)

                # Normalize
                angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                joint_range[0] = (180. + joint_range[0]) / 360.
                joint_range[1] = (180. + joint_range[1]) / 360.

            obs = np.concatenate([obs, [angle], joint_range])
            return obs

        # Skip body 0 (world) and collect observations for all other bodies
        full_obs = np.concatenate([_get_obs_per_limb(i) for i in range(1, self.model.nbody)])
        return full_obs.ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def render(self):
        return super().render()
