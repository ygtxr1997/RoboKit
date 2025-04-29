import copy
import numpy as np
from numpy.linalg import norm
from evdev import InputDevice, list_devices, ecodes
import select
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_SPACE, K_m
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import transforms3d
import transforms3d.quaternions as quat
from ahrs.filters import Madgwick


class IMUControl:
    def __init__(self, device_name="Pro Controller (IMU)",

                 ):
        self.device_name = device_name
        self.imu = self.find_imu(device_name)
        print(f'[IMUControl] Using {self.imu.path} ({self.imu.name})')

        # ---------- 初始化 ----------
        self.AXIS_MAP = {
            ecodes.ABS_RX: 'gx', ecodes.ABS_RY: 'gy', ecodes.ABS_RZ: 'gz',
            ecodes.ABS_X: 'ax', ecodes.ABS_Y: 'ay', ecodes.ABS_Z: 'az',
        }
        self.RAW2RAD = 2000.0 / 32767 * np.pi / 180
        self.RAW2ACC = 9.8 / 32767

        self.madgwick = Madgwick(sampleperiod=1 / 833)
        self.q_pose = np.array([1.0, 0.0, 0.0, 0.0])
        self.rpy_rel = np.array([0.0, 0.0, 0.0])

        self.state = {}
        self.t_prev = None

        # Calibration
        GYRO_SCALE, DEADZONE, LPF_ALPHA, bias_gyro = self.calibrate_imu()
        self.GYRO_SCALE = GYRO_SCALE
        self.DEADZONE = DEADZONE
        self.LPF_ALPHA = LPF_ALPHA
        self.bias_gyro = bias_gyro

        # 第一次滤波
        self.omega_filtered = np.zeros(3)
        self.omega_filtered_2nd = np.zeros(3)

    def calibrate_imu(self):
        imu = self.imu
        state = self.state
        AXIS_MAP = self.AXIS_MAP
        RAW2RAD = self.RAW2RAD
        RAW2ACC = self.RAW2ACC

        calib_samples = []

        print(f"[IMUControl] calibrating...")
        while len(calib_samples) < 800:
            r, _, _ = select.select([imu.fd], [], [], 0)
            if r:
                try:
                    for evt in imu.read():
                        if evt.type == ecodes.EV_ABS and evt.code in AXIS_MAP:
                            state[AXIS_MAP[evt.code]] = evt.value
                            if all(k in state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                                omega_now = np.array([state['gx'], state['gy'], state['gz']]) * RAW2RAD
                                accel_now = np.array([state['ax'], state['ay'], state['az']]) * RAW2ACC
                                calib_samples.append((omega_now, accel_now))
                except BlockingIOError:
                    pass

        gyro_samples = np.array([o for o, a in calib_samples])
        bias_gyro = np.mean(gyro_samples, axis=0)
        gyro_std = np.std(gyro_samples, axis=0)  # 每个轴的标准差 (shape=(3,))

        # 为每个轴分别设置 DEADZONE
        GYRO_SCALE = 0.03
        DEADZONE_XYZ = 0.03 * gyro_std * GYRO_SCALE  # 逐轴 deadzone
        LPF_ALPHA = np.clip(np.linalg.norm(gyro_std) * 20, 0.01, 0.1)

        print(f"[IMUControl][校准完成] DEADZONE={DEADZONE_XYZ}, LPF_ALPHA={LPF_ALPHA:.5f}, GYRO_SCALE={GYRO_SCALE}")
        return GYRO_SCALE, DEADZONE_XYZ, LPF_ALPHA, bias_gyro

    def capture_imu_pose(self):
        imu = self.imu
        madgwick = self.madgwick
        AXIS_MAP = self.AXIS_MAP
        state = self.state
        RAW2RAD = self.RAW2RAD
        RAW2ACC = self.RAW2ACC

        GYRO_SCALE = self.GYRO_SCALE
        LPF_ALPHA = self.LPF_ALPHA
        DEADZONE_XYZ = self.DEADZONE  # 注意现在是数组了
        bias_gyro = self.bias_gyro

        q_pose_old = self.q_pose.copy()
        q_pose_new = self.q_pose.copy()

        r, _, _ = select.select([imu.fd], [], [], 0)
        if r:
            try:
                for evt in imu.read():
                    if evt.type == ecodes.EV_ABS and evt.code in AXIS_MAP:
                        state[AXIS_MAP[evt.code]] = evt.value
                        ts = evt.timestamp()
                        t_now = ts if isinstance(ts, float) else ts[0] + ts[1] * 1e-6

                        if all(k in state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                            omega_raw = np.array([state['gx'], state['gy'], state['gz']]) * RAW2RAD
                            accel_raw = np.array([state['ax'], state['ay'], state['az']]) * RAW2ACC

                            omega = (omega_raw - bias_gyro) * GYRO_SCALE

                            # 第一次滤波
                            self.omega_filtered = (1.0 - LPF_ALPHA) * self.omega_filtered + LPF_ALPHA * omega

                            # 第二次滤波
                            if not hasattr(self, 'omega_filtered_2nd'):
                                self.omega_filtered_2nd = np.zeros(3)
                            self.omega_filtered_2nd = ((1.0 - LPF_ALPHA) * self.omega_filtered_2nd +
                                                       LPF_ALPHA * self.omega_filtered)

                            # 分别判断每个轴是否小于对应 deadzone
                            mask = np.abs(self.omega_filtered_2nd) < DEADZONE_XYZ
                            self.omega_filtered_2nd = self.omega_filtered_2nd * (~mask)

                            if self.t_prev is not None:
                                dt = t_now - self.t_prev
                                madgwick.sampleperiod = dt
                                q_pose_new = madgwick.updateIMU(copy.deepcopy(self.q_pose),
                                                                gyr=self.omega_filtered_2nd,
                                                                acc=accel_raw)
                                q_rel = quat.qmult(quat.qinverse(q_pose_old), q_pose_new)
                                rpy_rel = self.quat2euler(q_rel)
                                self.rpy_rel = rpy_rel

                            self.t_prev = t_now
            except BlockingIOError:
                print("Error: BlockingIOError")
            except Exception as e:
                print("Error: ", e)

        self.q_pose = q_pose_new
        return self.q_pose, self.rpy_rel

    @staticmethod
    def display_rpy(screen, rpy):
        """在 HUD 中显示 roll, pitch, yaw"""
        try:
            roll, pitch, yaw = rpy
            font = pygame.font.Font(None, 36)
            text = f"Roll: {np.degrees(roll):.2f}°  Pitch: {np.degrees(pitch):.2f}°  Yaw: {np.degrees(yaw):.2f}°"
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))
        except Exception as e:
            print(f"Error displaying RPY: {e}")

    @staticmethod
    def find_imu(name_keyword='Pro Controller (IMU)'):
        for path in list_devices():
            dev = InputDevice(path)
            if name_keyword in dev.name:
                return dev
        raise RuntimeError('IMU 未找到')

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
    def quat2mat4x4(q):
        try:
            m3 = quat.quat2mat(q)
            m4 = np.eye(4, dtype=np.float32)
            m4[:3, :3] = m3
            return m4
        except Exception as e:
            print(f"Error in quat2mat4x4: {e}")
            return np.eye(4, dtype=np.float32)

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
        glEnd()  # "gradio",

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

