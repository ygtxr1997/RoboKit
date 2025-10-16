import numpy as np
import time

import roslibpy


import roslibpy
import roslibpy.actionlib
from tqdm import tqdm
import time

from transforms3d.quaternions import qmult, qconjugate, qnorm
from transforms3d.quaternions import qmult, qconjugate, qnorm, mat2quat

from robokit.network.robot_client_piper import RobotClientPiper
from robokit.network.vr_handler import QuestHandler


client = roslibpy.Ros(host='192.168.5.243', port=9090) # Change host to the IP of the robot
client.run()

# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)

rc_piper = RobotClientPiper(client)

quest_handler = QuestHandler()
time.sleep(0.5)

right_gripper_pressed = False


def calculate_relative_quat(now_quat, last_quat, input_format='xyzw'):
    """
    计算两个四元数之间的相对四元数

    Parameters:
    -----------
    now_quat : array-like
        当前四元数，shape为(4,)
    last_quat : array-like
        上一个四元数，shape为(4,)
    input_format : str
        输入四元数的格式，'xyzw' 或 'wxyz'
        - 'xyzw': [x, y, z, w] 格式
        - 'wxyz': [w, x, y, z] 格式（transforms3d默认格式）

    Returns:
    --------
    relative_quat : numpy.ndarray
        相对四元数，格式与input_format一致
    """

    # 转换为numpy数组
    now_quat = np.array(now_quat, dtype=float)
    last_quat = np.array(last_quat, dtype=float)

    # 归一化四元数（确保是单位四元数）
    now_quat = now_quat / qnorm(now_quat)
    last_quat = last_quat / qnorm(last_quat)

    # 根据输入格式转换为transforms3d的wxyz格式
    if input_format == 'xyzw':
        # 从[x,y,z,w]转换为[w,x,y,z]
        now_quat_wxyz = np.array([now_quat[3], now_quat[0], now_quat[1], now_quat[2]])
        last_quat_wxyz = np.array([last_quat[3], last_quat[0], last_quat[1], last_quat[2]])
    elif input_format == 'wxyz':
        now_quat_wxyz = now_quat.copy()
        last_quat_wxyz = last_quat.copy()
    else:
        raise ValueError("input_format must be 'xyzw' or 'wxyz'")

    # 计算相对四元数: relative_quat = now_quat * conjugate(last_quat)
    last_quat_conj = qconjugate(last_quat_wxyz)
    relative_quat_wxyz = qmult(now_quat_wxyz, last_quat_conj)

    # 转换回原始格式
    if input_format == 'xyzw':
        # 从[w,x,y,z]转换为[x,y,z,w]
        relative_quat = np.array([
            relative_quat_wxyz[1],
            relative_quat_wxyz[2],
            relative_quat_wxyz[3],
            relative_quat_wxyz[0]
        ])
    else:
        relative_quat = relative_quat_wxyz

    return relative_quat


def calculate_next_quat(relative_quat, now_quat, input_format='xyzw', rotation_order='global'):
    """
    根据相对四元数和当前四元数计算下一个四元数

    Parameters:
    -----------
    relative_quat : array-like
        相对旋转四元数，shape为(4,)
    now_quat : array-like
        当前四元数，shape为(4,)
    input_format : str
        输入四元数的格式，'xyzw' 或 'wxyz'
        - 'xyzw': [x, y, z, w] 格式
        - 'wxyz': [w, x, y, z] 格式（transforms3d默认格式）
    rotation_order : str
        旋转顺序，'local' 或 'global'
        - 'local': 相对旋转在当前坐标系下进行，next_quat = now_quat * relative_quat
        - 'global': 相对旋转在全局坐标系下进行，next_quat = relative_quat * now_quat

    Returns:
    --------
    next_quat : numpy.ndarray
        下一个四元数，格式与input_format一致
    """

    # 转换为numpy数组
    relative_quat = np.array(relative_quat, dtype=float)
    now_quat = np.array(now_quat, dtype=float)

    # 归一化四元数（确保是单位四元数）
    relative_quat = relative_quat / qnorm(relative_quat)
    now_quat = now_quat / qnorm(now_quat)

    # 根据输入格式转换为transforms3d的wxyz格式
    if input_format == 'xyzw':
        # 从[x,y,z,w]转换为[w,x,y,z]
        relative_quat_wxyz = np.array([relative_quat[3], relative_quat[0], relative_quat[1], relative_quat[2]])
        now_quat_wxyz = np.array([now_quat[3], now_quat[0], now_quat[1], now_quat[2]])
    elif input_format == 'wxyz':
        relative_quat_wxyz = relative_quat.copy()
        now_quat_wxyz = now_quat.copy()
    else:
        raise ValueError("input_format must be 'xyzw' or 'wxyz'")

    # 根据旋转顺序计算下一个四元数
    if rotation_order == 'local':
        # 局部旋转：在当前坐标系下应用相对旋转
        next_quat_wxyz = qmult(now_quat_wxyz, relative_quat_wxyz)
    elif rotation_order == 'global':
        # 全局旋转：在全局坐标系下应用相对旋转
        next_quat_wxyz = qmult(relative_quat_wxyz, now_quat_wxyz)
    else:
        raise ValueError("rotation_order must be 'local' or 'global'")

    # 转换回原始格式
    if input_format == 'xyzw':
        # 从[w,x,y,z]转换为[x,y,z,w]
        next_quat = np.array([
            next_quat_wxyz[1],
            next_quat_wxyz[2],
            next_quat_wxyz[3],
            next_quat_wxyz[0]
        ])
    else:
        next_quat = next_quat_wxyz

    return next_quat


def map_quat_coordinate_system(now_quat, input_format='xyzw'):
    """
    四元数坐标系映射，对应position的xyz->zxy映射

    Parameters:
    -----------
    now_quat : array-like
        输入四元数，shape为(4,)
    input_format : str
        输入四元数格式，'xyzw' 或 'wxyz'

    Returns:
    --------
    mapped_quat : numpy.ndarray
        映射后的四元数，格式与输入相同
    """
    now_quat = np.array(now_quat, dtype=float)
    now_quat = now_quat / qnorm(now_quat)

    # 坐标系变换矩阵：xyz -> zxy
    transform_matrix = np.array([
        [0., 0., 1.],  # 新x = 原z
        [1., 0., 0.],  # 新y = 原x
        [0., 1., 0.]  # 新z = 原y
    ])

    # 变换四元数
    transform_quat = mat2quat(transform_matrix)  # wxyz格式
    transform_quat_inv = qconjugate(transform_quat)

    # 转换为wxyz格式进行计算
    if input_format == 'xyzw':
        now_quat_wxyz = np.array([now_quat[3], now_quat[0], now_quat[1], now_quat[2]])
    else:
        now_quat_wxyz = now_quat.copy()

    # 应用变换：q_new = q_transform * q * q_transform^(-1)
    temp_quat = qmult(transform_quat, now_quat_wxyz)
    mapped_quat_wxyz = qmult(temp_quat, transform_quat_inv)

    # 转换回原格式
    if input_format == 'xyzw':
        return np.array([mapped_quat_wxyz[1], mapped_quat_wxyz[2],
                         mapped_quat_wxyz[3], mapped_quat_wxyz[0]])
    else:
        return mapped_quat_wxyz


quest_zero_xyz = None
quest_zero_quat = None
while True:
    # print(quest_handler.get_last_buttons())

    # Check tele-op activated
    # if not quest_handler.is_right_gripper_pressed() and not right_gripper_pressed:  # Do nothing
    #     right_gripper_pressed = False
    #     time.sleep(0.1)
    #     continue

    if not quest_handler.is_right_gripper_pressed():
        time.sleep(0.03)
        continue

    quest_latest_data = quest_handler.get_latest_euler()
    now_xyz = quest_latest_data['now_xyz']
    now_quat = quest_latest_data['quat']
    # print('now_xyz', now_xyz)

    # time.sleep(0.2)

    if quest_zero_xyz is None:
        quest_zero_xyz = np.array(now_xyz)
        quest_zero_quat = np.array(now_quat)
        continue

    now_xyz = np.array(now_xyz) - quest_zero_xyz
    now_quat = calculate_relative_quat(now_quat, quest_zero_quat)  # quest relative rotation

    mapped_xyz = np.array([now_xyz[2], now_xyz[0], now_xyz[1]])  # z,x,y
    mapped_quat = map_quat_coordinate_system(now_quat)

    xyz_offset = np.array([  0.274933, -0.030914, 0.255386])
    quat_offset = np.array([0,  0.8872565080156615, 0,  0.4612763694184371])

    mapped_xyz = xyz_offset + mapped_xyz * 0.5
    mapped_quat = calculate_next_quat(mapped_quat, quat_offset)

    # mapped_xyz = xyz_offset
    linear_message = {'dx': mapped_xyz[0], 'dy': mapped_xyz[1], 'dz': mapped_xyz[2]}
    angular_message = {'qx': mapped_quat[0], 'qy': mapped_quat[1], 'qz': mapped_quat[2], 'qw': mapped_quat[3]}
    rc_piper.moveit_endpose_send(
        linear_message,
        angular_message=angular_message,
    )
    print(now_xyz, quest_zero_xyz)
