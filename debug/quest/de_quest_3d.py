import sys
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_SPACE, K_m
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import transforms3d
import transforms3d.quaternions as quat
from transforms3d.euler import quat2euler, mat2euler, euler2quat
from transforms3d.quaternions import qmult, qinverse, quat2mat, axangle2quat

# 如果没有这个模块，注释掉这行
try:
    from robokit.network.vr_handler import QuestHandler
except ImportError:
    print("QuestHandler not available, using dummy data")
    class QuestHandler:
        def __init__(self):
            self.q_now = [1, 0, 0, 0]
            self.angle = 0
        def reset_pose(self):
            self.angle = 0
        def remap_xyz_swap_xz(self, q):
            return q
        def get_latest_euler(self):
            self.angle += 0.01
            return {
                'euler': [np.sin(self.angle)*0.5, np.cos(self.angle)*0.3, self.angle*0.1, 0, 0, 0],
                'quat': [np.cos(self.angle/2), np.sin(self.angle/2)*0.2, 0, 0]
            }


class OpenGLHelper(object):
    @staticmethod
    def remap_xyz_swap_xz(q):
        """X 和 Z 轴交换: 给定一个坐标系转换回应到新坐标系角度"""
        try:
            q_y_90 = quat.axangle2quat([0, 1, 0], np.pi / 2)
            q_z_90 = quat.axangle2quat([1, 0, 0], np.pi / 2)
            q_remap_zxy = quat.qmult(q_y_90, q_z_90)
            return quat.qmult(q_remap_zxy, q)
        except Exception as e:
            print(f"Error in remap_xyz_swap_xz: {e}")
            return q  # 返回原始四元数

    @staticmethod
    def remap_xyz_swap_yz(q):
        """Y 和 Z 轴交换: 给定一个坐标系转换到新坐标系角度"""
        try:
            # 绕X轴旋转90度来交换Y和Z轴
            q_x_90 = quat.axangle2quat([1, 0, 0], np.pi / 2)
            return quat.qmult(q_x_90, q)
        except Exception as e:
            print(f"Error in remap_xyz_swap_yz: {e}")
            return q  # 返回原始四元数

    @staticmethod
    def remap_xyz_swap_yz_negative(q):
        """Y 和 Z 轴交换(负向): 给定一个坐标系转换到新坐标系角度"""
        try:
            # 绕X轴旋转-90度来交换Y和Z轴(另一个方向)
            q_x_neg90 = quat.axangle2quat([1, 0, 0], -np.pi / 2)
            return quat.qmult(q_x_neg90, q)
        except Exception as e:
            print(f"Error in remap_xyz_swap_yz_negative: {e}")
            return q  # 返回原始四元数

    @staticmethod
    def quat2mat4x4(q):
        try:
            m3 = quat.quat2mat(q)
            m4 = np.eye(4, dtype=np.float32)
            m4[:3, :3] = m3
            return m4.flatten()  # 重要：OpenGL需要扁平化矩阵
        except Exception as e:
            print(f"Error in quat2mat4x4: {e}")
            return np.eye(4, dtype=np.float32).flatten()

    @staticmethod
    def quat2euler(q):
        rpy = transforms3d.euler.quat2euler(q, axes='sxyz')  # 四元数到欧拉角，输出 (roll, pitch, yaw)
        return rpy

    @staticmethod
    def draw_airplane():
        """绘制飞机模型"""
        glBegin(GL_QUADS)
        glColor3f(0.8, 0.8, 0.8)
        body = [(-0.1, 0.0, -0.6), (0.1, 0.0, -0.6), (0.1, 0.0, 0.6), (-0.1, 0.0, 0.6)]
        for v in body: glVertex3f(*v)
        glColor3f(0.2, 0.6, 1.0)
        wing = [(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0), (0.4, 0.0, 0.2), (-0.4, 0.0, 0.2)]
        for v in wing: glVertex3f(*v)
        glEnd()
        glBegin(GL_TRIANGLES)
        glColor3f(1.0, 0.2, 0.2)
        tail = [(-0.05, 0.0, -0.6), (0.05, 0.0, -0.6), (0.0, 0.2, -0.5)]
        for v in tail: glVertex3f(*v)
        glEnd()

    @staticmethod
    def draw_axes():
        """绘制坐标轴（X, Y, Z）"""
        glBegin(GL_LINES)

        # X 轴（红色）
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)

        # Y 轴（绿色）
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)

        # Z 轴（蓝色）
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)

        glEnd()

    @staticmethod
    def display_rpy(screen, rpy):
        """在 HUD 中显示 roll, pitch, yaw - 简化版本"""
        try:
            roll, pitch, yaw = rpy
            # 输出到控制台而不是屏幕，避免字体问题
            print(f"Roll: {np.degrees(roll):.1f}° Pitch: {np.degrees(pitch):.1f}° Yaw: {np.degrees(yaw):.1f}°", end='\r')
        except Exception as e:
            print(f"Error displaying RPY: {e}")


# 初始化
pygame.init()

# Mac特定设置 - 关键修复
if sys.platform == "darwin":  # macOS
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    # 不设置Core Profile，使用兼容性模式

W, H = 800, 600
screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
pygame.display.set_caption("IMU 相对控制 + 坐标映射 - Mac Fixed")

# 简化的OpenGL初始化
try:
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1)
    # 只有在支持的情况下才设置投影
    try:
        gluPerspective(60, W / H, 0.1, 100)
        glTranslatef(0, 0, -3)
    except:
        print("Using simplified OpenGL setup")
except Exception as e:
    print(f"OpenGL setup error: {e}")

imu_controller = QuestHandler()
use_remap = True

# ---------- 主循环 ----------
clock = pygame.time.Clock()
running = True
q_pose = imu_controller.q_now

try:
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    print("[参考姿态重置]")
                    imu_controller.reset_pose()
                if event.key == K_m:
                    use_remap = not use_remap
                    print(f"[坐标轴映射切换] 当前 {'启用' if use_remap else '关闭'}")

        # 获取数据
        try:
            euler_data = imu_controller.get_latest_euler()
            rpy_now = euler_data['now_euler']
            rpy_rel = euler_data['euler']
            q_now = euler_data['quat']
            q_now = OpenGLHelper.remap_xyz_swap_yz_negative(q_now) if use_remap else q_now
        except Exception as e:
            print(f"Data error: {e}")
            q_now = [1, 0, 0, 0]
            rpy_now = [0, 0, 0]

        # 渲染
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        ## 绘制坐标轴
        try:
            glPushMatrix()
            OpenGLHelper.draw_axes()
            glPopMatrix()
        except:
            OpenGLHelper.draw_axes()

        ## 绘制飞机
        try:
            glPushMatrix()
            glMultMatrixf(OpenGLHelper.quat2mat4x4(q_now))
            OpenGLHelper.draw_airplane()
            glPopMatrix()
        except Exception as e:
            print(f"Render error: {e}")
            OpenGLHelper.draw_airplane()

        ## 显示 RPY (简化版本)
        try:
            OpenGLHelper.display_rpy(screen, rpy_now)
        except:
            pass

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    print("\n[用户中断退出]")
finally:
    pygame.quit()
    sys.exit()