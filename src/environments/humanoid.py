#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       Several Walkers
#   @author:
#       Tingwu (Wilson) Wang, Nov. 22nd, 2017
# -----------------------------------------------------------------------------

import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from src.environments.mujoco_parser import quat2expmap
from gymnasium.spaces import Box


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        # "render_fps": 25,
    }
    def __init__(self, xml, idx, seed=None, **kwargs):
        self.xml = xml
        self.idx = idx
        render_mode = kwargs.get('render_mode', None)
        self._desired_render_mode = render_mode
        print(f"{self.xml=}")
        # get from _get_obs
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
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        terminated = bool(done)
        truncated = False
        if hasattr(reward, 'item'):
            reward = float(reward.item())
        else:
            reward = float(reward)
        return ob, reward, terminated, truncated, {}

    def _get_obs(self):
        # qpos = self.model.data.qpos
        # qvel = self.model.data.qvel
        # return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
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
        return full_obs.ravel().astype(np.float32)

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
