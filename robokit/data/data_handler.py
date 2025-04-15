import numpy as np


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
data_dict = {
    "primary_rgb": np.zeros((256, 256, 3), dtype=np.uint8),
    "gripper_rgb": np.zeros((256, 256, 3), dtype=np.uint8),
    "primary_depth": np.zeros((256, 256), dtype=np.float32),
    "gripper_depth": np.zeros((256, 256), dtype=np.float32),
    "language_text": "None",
    "actions": np.zeros((7,), dtype=np.float32),  # (x,y,z,row,pitch,yaw,g)
    "rel_actions": np.zeros((7,), dtype=np.float32),  # (dx,dy,dz,d_row,d_pitch,d_yaw,g)
    "robot_obs": np.zeros((15,), dtype=np.float32),
    # (tcp pos (3), tcp ori (3), gripper width (1), joint_states (6) in rad, gripper_action (1)
}

