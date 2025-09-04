import mujoco
import mujoco.viewer
import time
import threading
import numpy as np


class HumanoidToolController:
    def __init__(self, model):
        self.model = model
        self.tool_equipped = False

        # Action indices
        self.base_actuators = ['right_hip_y', 'right_knee', 'left_hip_y', 'left_knee',
                               'right_shoulder1', 'right_elbow', 'left_shoulder1', 'left_elbow']
        self.wrist_actuator = 'right_wrist'

        self.base_action_dim = len(self.base_actuators)
        self.total_action_dim = self.base_action_dim + 1  # +1 for wrist when tool equipped

    def get_action_space_dim(self):
        """Return current action space dimension"""
        return self.total_action_dim if self.tool_equipped else self.base_action_dim

    def apply_action(self, data, action):
        """Apply action with proper dimension handling"""
        # Apply base humanoid actions
        for i, actuator_name in enumerate(self.base_actuators):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            data.ctrl[actuator_id] = action[i]

        # Apply wrist action if tool is equipped
        if self.tool_equipped and len(action) > self.base_action_dim:
            wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.wrist_actuator)
            data.ctrl[wrist_id] = action[self.base_action_dim]
        else:
            # Zero out wrist when no tool
            wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.wrist_actuator)
            data.ctrl[wrist_id] = 0


def equip_tool_to_hand(model, data, controller):
    """Move tool to wrist and weld it"""

    # Get wrist body position
    wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist')
    wrist_pos = data.xpos[wrist_body_id].copy()

    # Position tool
    tool_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'tool_free')
    tool_qpos_start = model.jnt_qposadr[tool_joint_id]

    data.qpos[tool_qpos_start:tool_qpos_start + 3] = wrist_pos + np.array([0, 0, 0.15])
    data.qpos[tool_qpos_start + 3:tool_qpos_start + 7] = [1, 0, 0, 0]

    # Zero velocities
    tool_qvel_start = model.jnt_dofadr[tool_joint_id]
    data.qvel[tool_qvel_start:tool_qvel_start + 6] = 0

    mujoco.mj_forward(model, data)

    # Activate constraint
    constraint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
    model.eq_active0[constraint_id] = 1

    # Update controller state
    controller.tool_equipped = True
    print(f"Tool equipped! Action space now: {controller.get_action_space_dim()} dimensions")


def drop_tool(model, controller):
    """Drop tool and disable wrist control"""

    constraint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
    model.eq_active0[constraint_id] = 0

    controller.tool_equipped = False
    print(f"Tool dropped! Action space now: {controller.get_action_space_dim()} dimensions")


def auto_sequence(model, data, controller):
    time.sleep(5)
    equip_tool_to_hand(model, data, controller)
    time.sleep(5)
    drop_tool(model, controller)


def main():
    model = mujoco.MjModel.from_xml_path("assets/humanoid_tool.xml")
    data = mujoco.MjData(model)
    controller = HumanoidToolController(model)

    print(f"Initial action space: {controller.get_action_space_dim()} dimensions")
    print("Without tool: [shoulder, elbow] for each arm + legs")
    print("With tool: [shoulder, elbow, wrist] for right arm + left arm + legs")

    sequence_thread = threading.Thread(target=auto_sequence, args=(model, data, controller), daemon=True)
    sequence_thread.start()

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
