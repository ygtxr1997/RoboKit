import numpy as np

from robokit.data import DataHandler


"""
Dict,keys=dict_keys(['rel_actions', 'language', 'rgb_gripper', 'scene_obs', 'rgb_static', 'language_text', 'robot_obs', 'gen_static', 'gen_gripper', 'future_frame_diff'])
rel_actions,<class 'numpy.ndarray'>,shape=(10, 7)
language,<class 'numpy.ndarray'>,shape=(1024,)
rgb_gripper,<class 'numpy.ndarray'>,shape=(2, 84, 84, 3)
scene_obs,<class 'numpy.ndarray'>,shape=(2, 24)
rgb_static,<class 'numpy.ndarray'>,shape=(2, 200, 200, 3)
language_text:<class 'str'>,len=30
robot_obs,<class 'numpy.ndarray'>,shape=(2, 15)
"""
img_w = 1280
img_h = 720
data_dict = {
    "primary_rgb": (np.random.randn(img_h, img_w, 3) * 255).astype(np.uint8),
    "gripper_rgb": (np.random.randn(img_h, img_w, 3) * 255).astype(np.uint8),
    "primary_depth": (np.random.randn(img_h, img_w) * 255).astype(np.float32),
    "gripper_depth": (np.random.randn(img_h, img_w) * 255).astype(np.float32),
    "language_text": np.array("None"),
    "actions": np.random.randn(7),  # (x,y,z,row,pitch,yaw,g)
    "rel_actions": np.random.randn(7),  # (j_x,j_y,j_z,j_ax,j_ay,j_az,g)
    "robot_obs": np.random.randn(14),
    # (tcp pos (3), tcp ori (3), gripper width (1), joint_states (6) in rad, gripper_action (1)
}

# 创建一个 DataHandler 实例
file_path = 'data.npz'
handler = DataHandler(data_dict)

# 保存数据
handler.save(file_path)

# 读取数据
loaded_data = handler.load(file_path)
print("加载的数据显示：", loaded_data)

# 获取指定数据项
actions_data = handler.get("actions")
print("获取到的 actions 数据：", actions_data)