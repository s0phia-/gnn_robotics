import numpy as np
import time
import mujoco
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


class HumanoidToolEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        # "render_fps": 25,
    }

    def __init__(self, **kwargs):
        self.tool_equipped = False

        super().__init__(
            model_path="./assets/humanoid_tool.xml",
            frame_skip=5,
            observation_space=None,
            **kwargs
        )

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    def _set_action_space(self):
        """Create 1D action space: [actuators..., pickup_tool, drop_tool]"""
        # Get actuator bounds
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        actuator_low, actuator_high = bounds.T

        # Add 2 extra actions for tool control (binary 0/1)
        tool_low = np.array([0.0, 0.0])  # [pickup, drop]
        tool_high = np.array([1.0, 1.0])

        # Concatenate actuator bounds with tool control bounds
        low = np.concatenate([actuator_low, tool_low])
        high = np.concatenate([actuator_high, tool_high])

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def step(self, action):
        """Process 1D action: action[:-2] = actuators, action[-2:] = [pickup, drop]"""
        # Extract actuator actions (all except last 2)
        actuator_actions = action[:-2]

        # Extract tool actions (last 2)
        pickup_action = action[-2]
        drop_action = action[-1]

        # Handle tool pickup/drop - but only ONE action per step
        if pickup_action > 0.5 and not self.tool_equipped:
            print("Attempting to equip tool...")
            self._equip_tool()
        elif drop_action > 0.5 and self.tool_equipped:
            print("Attempting to drop tool...")
            self._drop_tool()

        # Determine which actuator actions to use based on tool state
        if self.tool_equipped:
            ctrl = actuator_actions
        else:
            ctrl = actuator_actions.copy()
            try:
                wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_wrist')
                ctrl[wrist_id] = 0.0
            except:
                pass  # Wrist actuator might not exist

        # Apply actions
        self.do_simulation(ctrl, self.frame_skip)

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _equip_tool(self):
        """Equip tool to hand using only constraint activation"""
        # Don't manually position the tool - let the constraint handle it

        # Just activate the constraint and let MuJoCo handle positioning
        constraint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
        self.model.eq_active0[constraint_id] = 1

        # Step physics a few times to let constraint settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.tool_equipped = True
        print("Tool constraint activated!")

    def _drop_tool(self):
        """Drop tool"""
        constraint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
        self.model.eq_active0[constraint_id] = 0
        self.tool_equipped = False

    def _get_obs(self):
        """Get observation including tool state"""
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        tool_state = np.array([float(self.tool_equipped)])

        return np.concatenate([position, velocity, tool_state])

    def reset_model(self):
        """Reset environment"""
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1

        self.set_state(qpos, qvel)
        self._drop_tool()

        return self._get_obs()

    def _get_reward(self):
        """Basic reward function"""
        height = self.data.qpos[2]
        upright_reward = height
        tool_reward = 1.0 if self.tool_equipped else 0.0
        return upright_reward + tool_reward

    def _get_terminated(self):
        """Episode termination condition"""
        height = self.data.qpos[2]
        return bool(height < 1.0)

    def _get_truncated(self):
        return False

    def _get_info(self):
        return {
            'tool_equipped': self.tool_equipped,
            'height': self.data.qpos[2]
        }


# Example usage:
if __name__ == "__main__":
    env = HumanoidToolEnv(render_mode="human")

    obs, info = env.reset()
    env.render()

    # Example action - pickup tool
    action = np.zeros(env.action_space.shape[0])
    action[-2] = 1.0  # Pickup tool

    print("Sending pickup command...")
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(f"Tool equipped after pickup: {info['tool_equipped']}")

    # Keep stepping with neutral actions to see the tool attached
    for i in range(250):
        action = np.zeros(env.action_space.shape[0])
        # No pickup/drop actions, just neutral movement
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if i == 0:  # Print status after first neutral step
            print(f"Tool still equipped: {info['tool_equipped']}")

        if terminated:
            obs, info = env.reset()

    # Now drop the tool
    action = np.zeros(env.action_space.shape[0])
    action[-1] = 1.0  # Drop tool

    print("Sending drop command...")
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(f"Tool equipped after drop: {info['tool_equipped']}")

    env.close()



#
#
#
#
# class HumanoidToolController:
#     def __init__(self, model):
#         self.model = model
#         self.tool_equipped = False
#
#         # Action indices
#         self.base_actuators = ['right_hip_y', 'right_knee', 'left_hip_y', 'left_knee',
#                                'right_shoulder1', 'right_elbow', 'left_shoulder1', 'left_elbow']
#         self.wrist_actuator = 'right_wrist'
#
#         self.base_action_dim = len(self.base_actuators)
#         self.total_action_dim = self.base_action_dim + 1  # +1 for wrist when tool equipped
#
#     def get_action_space_dim(self):
#         """Return current action space dimension"""
#         return self.total_action_dim if self.tool_equipped else self.base_action_dim
#
#     def apply_action(self, data, action):
#         """Apply action with proper dimension handling"""
#         # Apply base humanoid actions
#         for i, actuator_name in enumerate(self.base_actuators):
#             actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
#             data.ctrl[actuator_id] = action[i]
#
#         # Apply wrist action if tool is equipped
#         if self.tool_equipped and len(action) > self.base_action_dim:
#             wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.wrist_actuator)
#             data.ctrl[wrist_id] = action[self.base_action_dim]
#         else:
#             # Zero out wrist when no tool
#             wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.wrist_actuator)
#             data.ctrl[wrist_id] = 0
#
#
# def equip_tool_to_hand(model, data, controller):
#     """Move tool to wrist and weld it"""
#
#     # Get wrist body position
#     wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist')
#     wrist_pos = data.xpos[wrist_body_id].copy()
#
#     # Position tool
#     tool_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'tool_free')
#     tool_qpos_start = model.jnt_qposadr[tool_joint_id]
#
#     data.qpos[tool_qpos_start:tool_qpos_start + 3] = wrist_pos + np.array([0, 0, 0.15])
#     data.qpos[tool_qpos_start + 3:tool_qpos_start + 7] = [1, 0, 0, 0]
#
#     # Zero velocities
#     tool_qvel_start = model.jnt_dofadr[tool_joint_id]
#     data.qvel[tool_qvel_start:tool_qvel_start + 6] = 0
#
#     mujoco.mj_forward(model, data)
#
#     # Activate constraint
#     constraint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
#     model.eq_active0[constraint_id] = 1
#
#     # Update controller state
#     controller.tool_equipped = True
#
#
# def drop_tool(model, controller):
#     """Drop tool and disable wrist control"""
#
#     constraint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
#     model.eq_active0[constraint_id] = 0
#
#     controller.tool_equipped = False
#
#
# def auto_sequence(model, data, controller):
#     time.sleep(5)
#     equip_tool_to_hand(model, data, controller)
#     time.sleep(5)
#     drop_tool(model, controller)
#
#
# def main():
#     model = mujoco.MjModel.from_xml_path("assets/humanoid_tool.xml")
#     data = mujoco.MjData(model)
#     controller = HumanoidToolController(model)
#
#     print(f"Initial action space: {controller.get_action_space_dim()} dimensions")
#     print("Without tool: [shoulder, elbow] for each arm + legs")
#     print("With tool: [shoulder, elbow, wrist] for right arm + left arm + legs")
#
#     sequence_thread = threading.Thread(target=auto_sequence, args=(model, data, controller), daemon=True)
#     sequence_thread.start()
#
#     mujoco.viewer.launch(model, data)
#
#
# if __name__ == "__main__":
#     main()
