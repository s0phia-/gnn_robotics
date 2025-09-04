import mujoco
import mujoco.viewer
import time
import threading
import numpy as np


def equip_tool_to_hand(model, data):
    """Move tool to hand and weld it"""

    # Get hand body position
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_body')
    hand_pos = data.xpos[hand_body_id].copy()

    # Position tool so its bottom end will be at hand when constraint activates
    tool_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'tool_free')
    tool_qpos_start = model.jnt_qposadr[tool_joint_id]

    # Tool center should be 0.15 above hand (so bottom touches hand)
    data.qpos[tool_qpos_start:tool_qpos_start + 3] = hand_pos + np.array([0, 0, 0.15])
    data.qpos[tool_qpos_start + 3:tool_qpos_start + 7] = [1, 0, 0, 0]

    # Zero velocities
    tool_qvel_start = model.jnt_dofadr[tool_joint_id]
    data.qvel[tool_qvel_start:tool_qvel_start + 6] = 0

    mujoco.mj_forward(model, data)

    # Activate constraint
    constraint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
    model.eq_active0[constraint_id] = 1


def drop_tool(model):
    """Drop the tool by deactivating the constraint"""

    constraint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'tool_constraint')
    model.eq_active0[constraint_id] = 0


def auto_sequence(model, data):
    time.sleep(5)
    equip_tool_to_hand(model, data)
    time.sleep(5)
    drop_tool(model)


def main():
    model = mujoco.MjModel.from_xml_path("assets/humanoid_tool.xml")
    data = mujoco.MjData(model)

    sequence_thread = threading.Thread(target=auto_sequence, args=(model, data), daemon=True)
    sequence_thread.start()

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()