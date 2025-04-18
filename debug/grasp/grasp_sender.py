#!/usr/bin/env python3
import numpy as np
from inovopy.robot import InovoRobot
from inovopy.geometry.transform import Transform
from inovopy.geometry import transform
from inovopy.logger import Logger
import re
import pprint


class GraspPoseSender(object):
    def __init__(self, robot_ip="192.168.1.7"):
        bot = InovoRobot.default_iva(robot_ip)
        bot.set_param(speed=100, accel=100, tcp_speed_linear=100, tcp_speed_angular=70)
        self.bot = bot

    def set_p(self, gripper_pose):
        R_final_euler = transform.mat_to_euler(gripper_pose[:3, :3])
        Target_cy = Transform(vec_mm=gripper_pose[:3, 3], euler_deg=R_final_euler)
        self.bot.linear(Target_cy)

    def run(self, objects):
        """
        Example data:
        objects = [
            {
                "name": "圆柱",
                "R": np.array([
                    [-0.08158159, 0.25795618, 0.96270615],
                    [0., 0.9659259, -0.2588189],
                    [-0.99666667, -0.02111486, -0.07880177]
                ]),
                "t": np.array([-0.0151532, 0.48732078, 0.08669137])
            },
            {
                "name": "球体",
                "R": np.array([
                    [0.14391817, 0.31790584, 0.9371358],
                    [-0.2771057, 0.92205524, -0.27023423],
                    [-0.95, -0.22079404, 0.22079402]
                ]),
                "t": np.array([0.03989823, 0.4437075, 0.0459725])
            }
        ]
        """
        pprint.pprint(objects)
        HOME = Transform(vec_mm=(0, 400, 450), euler_deg=(180, 0, 0))

        # 启动机械手
        bot = self.bot
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
            self.set_p(T_final_gripper)
            bot.sleep(0.5)

            # 靠近物体（离物体10cm）
            T_D10 = np.eye(4)
            T_D10[:3, 3] = np.array([0, 0, -220])  # 调整到更近的距离 #-230
            T_p2 = T_gp @ T_ry90 @ T_rz90 @ T_D10
            self.set_p(T_p2)
            bot.sleep(0.5)

            # 抓取物体
            bot.gripper_set("close")
            bot.sleep(0.5)

            # 举升物体（Z轴抬高10cm）
            T_p3 = T_p2.copy()
            T_p3[2, 3] += 50
            self.set_p(T_p3)
            bot.sleep(0.5)

            # 向X轴反方向移动20cm
            T_p4 = T_p3.copy()
            T_p4[0, 3] -= 200  # X轴减去200mm
            self.set_p(T_p4)
            bot.sleep(0.5)

            # 放下物体（Z轴下降10cm）
            T_p5 = T_p4.copy()
            T_p5[2, 3] -= 50
            self.set_p(T_p5)
            bot.sleep(0.5)

            # 放置物体
            bot.gripper_set("open")
            bot.sleep(0.5)

            # 回到安全高度
            T_p6 = T_p5.copy()
            T_p6[2, 3] += 100
            self.set_p(T_p6)
            bot.linear(HOME)
            # bot.sleep(1)

        print("所有物体处理完成！")
        # bot.linear(HOME)
        self.bot.sleep(3)
        self.bot.disconnect()