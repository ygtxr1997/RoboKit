#!/usr/bin/env python3
import numpy as np
from inovopy.robot import InovoRobot
from inovopy.geometry.transform import Transform
from inovopy.geometry import transform
from inovopy.logger import Logger
import re
import pprint

def parse_terminal_output(input_str):
    # 将输入按行分割
    lines = input_str.split('\n')
    objects = []
    i = 0
    current_id = None
    
    # 遍历每一行
    while i < len(lines):
        line = lines[i].strip()
        
        # 提取物体编号
        if line.startswith("Processing end_points number:"):
            current_id = int(line.split(':')[-1].strip())
            i += 1
        
        # 提取抓取位姿
        elif line == "grasp pose for visualize is:":
            # 提取旋转矩阵（3 行）
            rotation_matrix = []
            for j in range(3):
                # r_line = lines[i + 1 + j].strip('[]')  # 去掉方括号
                # nums = r_line.split()  # 分割成数字列表
                nums = re.findall(r'-?\d+\.?\d*', line)
                rotation_matrix.append([float(num) for num in nums])  # 转换为浮点数
            
            # 提取平移向量（1 行）
            trans_line = lines[i + 4].strip('[]')  # 去掉方括号
            translation_vector = [float(num) for num in trans_line.split()]  # 转换为浮点数
            
            # 创建字典
            obj = {
                "name": f"ID{current_id}",
                "R": np.array(rotation_matrix),
                "t": np.array(translation_vector)
            }
            objects.append(obj)
            
            # 跳过已处理的行（1行标题 + 3行旋转 + 1行平移）
            i += 5
        
        # 其他行跳过
        else:
            i += 1
    
    return objects

bot = InovoRobot.default_iva("192.168.1.7")
bot.set_param(speed=100, accel=100, tcp_speed_linear=100, tcp_speed_angular=70)

def set_p(gripper_pose):
    R_final_euler = transform.mat_to_euler(gripper_pose[:3,:3])
    Target_cy = Transform(vec_mm=gripper_pose[:3,3], euler_deg=R_final_euler)
    bot.linear(Target_cy)

# 定义抓取物体列表（示例数据）
# objects = [
#     {
#         "name": "球体",
#         "R": np.array([
#                 [ 0.14391817 , 0.31790584 , 0.9371358 ],
#                 [-0.2771057 ,  0.92205524 ,-0.27023423],
#                 [-0.95     ,  -0.22079404 , 0.22079402]
#         ]),
#         "t": np.array([-0.02327531 , 0.46720484,  0.08289546])

#     },
#     {
#         "name": "圆柱",
#         "R": np.array([
#             [ 0.3237263,  -0.8820168,   0.34241456],
#             [ 0.1336295,   0.40089735,  0.90632486],
#             [-0.93666667, -0.24764448,  0.24764447]
#         ]),
#         "t": np.array([0.02901126, 0.5341057,  0.03871423])
#     },
#     {
#         "name": "冰块",
#         "R": np.array([
#             [ 0.3237263,  -0.8820168,   0.34241456],
#             [ 0.1336295,   0.40089735,  0.90632486],
#             [-0.93666667, -0.24764448,  0.24764447]
#         ]),
#         "t": np.array([0.02901126, 0.5341057,  0.03871423])
#     }
# ]

input_str = """(graspnet) (base) qqy@qpc:~/workspace/graspnet/graspnet-baseline$ python demo_mesh.py 
WARNING:root:Failed to import geometry msgs in rigid_transformations.py.
WARNING:root:Failed to import ros dependencies in rigid_transforms.py
WARNING:root:autolab_core not installed as catkin package, RigidTransform ros methods will be unavailable
-> loaded checkpoint logs/log_rs/checkpoint.tar (epoch: 18)
Processing end_points number: 0
Retry 1 for end_points 0
Retry 2 for end_points 0
Grasp: score:0.5250277519226074, width:0.07416253536939621, height:0.019999999552965164, depth:0.029999999329447746, translation:[-0.03752688  0.53950715  0.03976621]
rotation:
[[ 0.3237263  -0.8820168   0.34241456]
 [ 0.1336295   0.40089735  0.90632486]
 [-0.93666667 -0.24764448  0.24764447]]
object id:-1
grasp pose for visualize is:
[[ 0.3237263  -0.8820168   0.34241456]
 [ 0.1336295   0.40089735  0.90632486]
 [-0.93666667 -0.24764448  0.24764447]]
[-0.03752688  0.53950715  0.03976621]
Processing end_points number: 1
Grasp: score:0.23785778880119324, width:0.08601535856723785, height:0.019999999552965164, depth:0.029999999329447746, translation:[0.06263897 0.4889151  0.0406348 ]
rotation:
[[-0.11794727 -0.89446706  0.431297  ]
 [-0.52428746  0.42495117  0.737929  ]
 [-0.8433333  -0.1390869  -0.5190797 ]]
object id:-1
grasp pose for visualize is:
[[-0.11794727 -0.89446706  0.431297  ]
 [-0.52428746  0.42495117  0.737929  ]
 [-0.8433333  -0.1390869  -0.5190797 ]]
[0.06263897 0.4889151  0.0406348 ]
Processing end_points number: 2
Retry 1 for end_points 2
Retry 2 for end_points 2
Retry 3 for end_points 2
Retry 4 for end_points 2
Retry 5 for end_points 2
Retry 6 for end_points 2
Grasp: score:0.12048748135566711, width:0.04676583409309387, height:0.019999999552965164, depth:0.019999999552965164, translation:[-0.06374837  0.47066727  0.01525619]
rotation:
[[-0.15578863 -0.58730066  0.79423416]
 [-0.33291125  0.7882278   0.51755875]
 [-0.93       -0.18377969 -0.3183159 ]]
object id:-1
grasp pose for visualize is:
[[-0.15578863 -0.58730066  0.79423416]
 [-0.33291125  0.7882278   0.51755875]
 [-0.93       -0.18377969 -0.3183159 ]]
[-0.06374837  0.47066727  0.01525619]"""

# 解析并生成 objects 列表
# objects = parse_terminal_output(input_str)
# objects = [
#     {
#         "name": "球体",
#         "R": np.array([
#             [ 0.3237263,  -0.8820168,   0.34241456],
#             [ 0.1336295,   0.40089735,  0.90632486],
#             [-0.93666667, -0.24764448,  0.24764447]
#         ]),
#         "t": np.array([-0.03752688,  0.53950715,  0.03976621])
#     },
#     {
#         "name": "圆柱",
#         "R": np.array([
#             [-0.11794727, -0.89446706,  0.431297  ],
#             [-0.52428746,  0.42495117,  0.737929  ],
#             [-0.8433333,  -0.1390869,  -0.5190797 ]
#         ]),
#         "t": np.array([0.06263897, 0.4889151,  0.0406348 ])
#     },
#     {
#         "name": "冰块",
#         "R": np.array([
#             [-0.15578863, -0.58730066,  0.79423416],
#             [-0.33291125,  0.7882278,   0.51755875],
#             [-0.93      , -0.18377969, -0.3183159 ]
#         ]),
#         "t": np.array([-0.06374837,  0.47066727,  0.01525619])
#     }
# ]
# objects = [
#     {
#         "name": "圆柱",
#         "R": np.array([
#             [-0.11794727  ,0.8207478  , 0.5589826 ],
#             [-0.52428746 , 0.42658958 ,-0.73698306],
#             [-0.8433333 , -0.37999272 , 0.3799927 ]
#         ]),
#         "t": np.array([-0.00992961 , 0.47549155 , 0.08026087])
#     },
# ]

objects = [
    {
        "name": "圆柱",
        "R": np.array([
            [-0.08158159,  0.25795618 , 0.96270615],
            [ 0.  ,        0.9659259,  -0.2588189 ],
            [-0.99666667, -0.02111486, -0.07880177]
        ]),
        "t": np.array([-0.0151532 ,  0.48732078 , 0.08669137])
    },
    {
        "name": "球体",
        "R": np.array([
            [ 0.14391817 , 0.31790584,  0.9371358 ],
            [-0.2771057 ,  0.92205524 ,-0.27023423],
            [-0.95     ,  -0.22079404 , 0.22079402]
        ]),
        "t": np.array([0.03989823 ,0.4437075  ,0.0459725 ])
    }
]


pprint.pprint(objects)
HOME = Transform(vec_mm=(0, 400, 450), euler_deg=(180, 0, 0))


# 启动机械手
bot.gipper_activate()
bot.linear(HOME)
bot.sleep(0.5)



# 定义通用变换矩阵
R_ry90 = transform.euler_to_mat(np.array([0, 90, 0]))
T_ry90 = np.eye(4)
T_ry90[:3, :3] = R_ry90

R_rz90 = transform.euler_to_mat(np.array([0, 0, -90]))
T_rz90 = np.eye(4)
T_rz90[:3, :3] = R_rz90

for obj in objects:
    print(f"正在处理物体: {obj['name']}")
    
    # 构建抓取位姿
    T_gp = np.eye(4)
    T_gp[:3, :3] = obj["R"]
    T_gp[:3, 3] = obj["t"] * 1000  # 转换为毫米
    
    # 预抓取位置（离物体25cm）
    T_D25 = np.eye(4)
    T_D25[:3, 3] = np.array([0, 0, -240])
    T_final_gripper = T_gp @ T_ry90 @ T_rz90 @ T_D25
    set_p(T_final_gripper)
    bot.sleep(0.5)

    # 靠近物体（离物体10cm）
    T_D10 = np.eye(4)
    T_D10[:3, 3] = np.array([0, 0, -220])  # 调整到更近的距离 #-230
    T_p2 = T_gp @ T_ry90 @ T_rz90 @ T_D10
    set_p(T_p2)
    bot.sleep(0.5)

    # 抓取物体
    bot.gripper_set("close")
    bot.sleep(0.5)

    # 举升物体（Z轴抬高10cm）
    T_p3 = T_p2.copy()
    T_p3[2, 3] += 50
    set_p(T_p3)
    bot.sleep(0.5)

    # 向X轴反方向移动20cm
    T_p4 = T_p3.copy()
    T_p4[0, 3] -= 200  # X轴减去200mm
    set_p(T_p4)
    bot.sleep(0.5)

    # 放下物体（Z轴下降10cm）
    T_p5 = T_p4.copy()
    T_p5[2, 3] -= 50
    set_p(T_p5)
    bot.sleep(0.5)

    # 放置物体
    bot.gripper_set("open")
    bot.sleep(0.5)

    # 回到安全高度
    T_p6 = T_p5.copy()
    T_p6[2, 3] += 100
    set_p(T_p6)
    bot.linear(HOME)
    # bot.sleep(1)

print("所有物体处理完成！")
# bot.linear(HOME)
bot.sleep(3)
bot.disconnect()