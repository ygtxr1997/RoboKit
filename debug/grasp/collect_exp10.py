#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import os
import time
import math
import numpy as np
from commander_api.motion_control_client import MotionControlClient, Waypoint, Motion
import tf
# The commander_api is defined at inovo_ws
import pyrealsense2 as rs

# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
profile = pipeline.start(config)
rospy.init_node("get_robot_state")
mc = MotionControlClient("default_move_group")



class DataCollector:
    def __init__(self):
        # rospy.init_node('data_collector', anonymous=True)
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        # self.pose_sub = rospy.Subscriber("/default_move_group/tcp_pose", PoseStamped, self.pose_callback)
        self.image = None
        self.pose = None
        self.count = 0
        self.max_count = 400
        self.save_interval = 0.5  # seconds

    # def image_callback(self, data):
    #     # try:
    #     #     self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #     # except Exception as e:
    #     #     print(e)
    #     frames = pipeline.wait_for_frames()
    #     color_frame = frames.get_color_frame()
    #     # 将图像转换为numpy数组
    #     self.color_image = np.asanyarray(color_frame.get_data())

    # def pose_callback(self, data):
    #     self.pose = data

    def save_data(self):
        os.makedirs("exp10", exist_ok=True)
        # if self.image is not None and self.pose is not None:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        image_filename = f"exp10/exp_{self.count}.jpg"
        depth_image_filename = f"exp9/exp_{self.count}_depth.png"
        pose_filename = f"exp10/exp_{self.count}.txt"
        cv2.imwrite(image_filename, color_image)
        cv2.imwrite(depth_image_filename, depth_image)

        # 将深度图转换为彩色图像以便于可视化
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_PLASMA)
        depth_colormap_filename = f"exp10/exp_{self.count}_depth_color.png"
        cv2.imwrite(depth_colormap_filename, depth_colormap)

        tcp_pose = mc.get_tcp_pose()

        # Pg2b = [tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z]
        # quaternion = [
        #         tcp_pose.orientation.x,
        #         tcp_pose.orientation.y,
        #         tcp_pose.orientation.z,
        #         tcp_pose.orientation.w
        #     ]


        with open(pose_filename, 'w') as f:
            f.write(f"Position: {tcp_pose.position}\n")
            f.write(f"Orientation: {tcp_pose.orientation}\n")

        print(f"Saved {image_filename} and {pose_filename}")
        self.count += 1


    def run(self):
        # Define the sphere param
        s_radius = 0.25 #15cm
        # s_center = [0.0, 0.40, 0.10]
        s_center = [0.0, 0.50, 0.0]
        num_r_pitch = 12
        num_r_yaw = 10
        camera_pitch = math.pi/2
        camera_yaw = 0
        camera_roll = 0
        flag_clockwise = True
        height_add = 0

        # rospy.init_node("picture_collector")

        # Traverse the points on the sphere surface, in polar axis
        for r_pitch in range(6, num_r_pitch-1):
            height_add += 0
            for r_yaw in range(num_r_yaw):

                # 摄像机原点加上这个平移矩阵就是抓手末端的原点，这里还涉及一个gripper坐标系
                # FRD （frount right down） = [ -33.95, 47.5, 0]
                # 后来又做了手眼标定，这里就不用再管了
                
                camera_pitch = ((num_r_pitch - r_pitch) / num_r_pitch * math.pi/2)
                # Avoid the xy desk plane
                if flag_clockwise:
                    camera_yaw = (r_yaw / num_r_yaw * math.pi * 2) - math.pi
                else:
                    camera_yaw = -1 * (r_yaw / num_r_yaw * math.pi * 2) - math.pi
                camera_yaw = camera_yaw + (math.pi) / (num_r_yaw)
                camera_roll = 0

                x =  s_radius * np.sin(camera_pitch) * np.sin(camera_yaw)
                y =  -s_radius * np.sin(camera_pitch) * np.cos(camera_yaw)
                z =  s_radius * np.cos(camera_pitch)
                # camera_pitch = (-math.pi/(2*num_r_pitch))+camera_pitch + math.pi/2 #frome gripper pose to camera pose
                camera_pitch = camera_pitch + math.pi/2 #最开始的侧面放摄像头
                # camera_pitch = camera_pitch + math.pi #它的效果是机械臂末端直接对着球心
                pose = [x, y, z]
                pose[0]=pose[0]+s_center[0]
                pose[1]=pose[1]+s_center[1]
                pose[2]=pose[2]+s_center[2] + height_add
                # pose = point + s_center
                # print(point)

                # direction = -point / np.linalg.norm(point)

                # transform_matrix = self.create_transform_matrix(np.array(pose), direction, s_center)
                # print("Transform matrix from point frame to s_center frame:")
                # print(transform_matrix)

                # Copied from inovo_ws motion_advanced.
                mc = MotionControlClient("default_move_group")

                ACCEL = 0.2
                VEL = 0.2
                BLENDL = 0.1

                print(" x: ", format(pose[0], '.4f'), " y: ", format(pose[1], '.4f'), " z: ", format(pose[2], '.4f'), " camera_pitch: ", format(camera_pitch*180/math.pi, '.4f'), " camera_yaw: ", format(camera_yaw*180/math.pi, '.4f'), " camera_roll: ", format(camera_roll*180/math.pi))
                quat = tf.transformations.quaternion_from_euler(camera_pitch, camera_yaw, 0)
                # camera_pitch=math.pi
                # camera_yaw=0
                # camera_roll=0

                m = Motion()

                #We should double check the defining of the rotation notation in robotic arm
                m.add(Waypoint(pose[0], pose[1], pose[2], camera_pitch, camera_roll, camera_yaw) \
                    .constrain_joint_acceleration(ACCEL) \
                    .constrain_joint_velocity(VEL) \
                    .set_blend(BLENDL, 0.5) \
                    .set_linear())
                try:
                    mc.run(m)
                    time.sleep(0.5)
                    self.save_data()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"An error occurred: {e}")
            flag_clockwise = not flag_clockwise



if __name__ == '__main__':
    try:
        collector = DataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass