import sys
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_SPACE, K_m
from OpenGL.GL import *
from OpenGL.GLU import *

from robokit.network.imu_control import IMUControl


imu_controller = IMUControl()

pygame.init()
W, H = 800, 600
screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
pygame.display.set_caption("IMU 相对控制 + 坐标映射")

glEnable(GL_DEPTH_TEST)
glClearColor(0.1, 0.1, 0.1, 1)
gluPerspective(60, W / H, 0.1, 100)
glTranslatef(0, 0, -3)

use_remap = True

# ---------- 主循环 ----------
clock = pygame.time.Clock()
running = True
q_pose = imu_controller.q_pose
try:
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    q_ref = q_pose.copy()
                    print("[参考姿态重置]")
                if event.key == K_m:
                    use_remap = not use_remap
                    print(f"[坐标轴映射切换] 当前 {'启用' if use_remap else '关闭'}")

        q_now, rpy_rel = imu_controller.capture_imu_pose()
        q_now = imu_controller.remap_xyz_swap_xz(q_now) if use_remap else q_now

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        ## 绘制坐标轴
        glPushMatrix()
        imu_controller.draw_axes()
        glPopMatrix()

        ## 绘制飞机
        glPushMatrix()
        glMultMatrixf(imu_controller.quat2mat4x4(q_now))
        imu_controller.draw_airplane()
        glPopMatrix()

        ## 显示 RPY
        print(rpy_rel)
        roll, pitch, yaw = rpy_rel
        imu_controller.display_rpy(screen, rpy_rel)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    print("\n[用户中断退出]")
finally:
    pygame.quit()
    sys.exit()
