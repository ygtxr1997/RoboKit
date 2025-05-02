import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
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
from transforms3d.euler import quat2euler, mat2euler, euler2quat
from transforms3d.quaternions import qmult, qinverse, quat2mat, axangle2quat
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


# class RawIMUHandler:
#     def __init__(self, device_name="Pro Controller (IMU)"
#                  ):
#         self.device_name = device_name
#         self.imu = self.find_imu(device_name)
#         print(f'[RawIMUHandler] Using {self.imu.path} ({self.imu.name})')
#
#         # ---------- 初始化 ----------
#         self.AXIS_MAP = {
#             ecodes.ABS_RX: 'gx', ecodes.ABS_RY: 'gy', ecodes.ABS_RZ: 'gz',
#             ecodes.ABS_X: 'ax', ecodes.ABS_Y: 'ay', ecodes.ABS_Z: 'az',
#         }
#         self.RAW2RAD = 2000.0 / 32767 * np.pi / 180
#         self.RAW2ACC = 9.8 / 32767
#
#         self.GYRO_SCALE = 1 / 16.4  # °/s
#         self.GYRO_TO_RAD = np.pi / 180
#         self.ACC_SCALE = 9.8 * 8.0 / 32767  # m/s²
#
#         self.madgwick = Madgwick(sampleperiod=1 / 833)
#         self.q_pose = np.array([1.0, 0.0, 0.0, 0.0])
#         self.rpy_rel = np.array([0.0, 0.0, 0.0])
#
#         self.state = {'ax': 0, 'ay': 0, 'az': 0, 'gx': 0, 'gy': 0, 'gz': 0}
#         self.state_converted = copy.deepcopy(self.state)
#         self.t_prev = None
#
#         # Calibration
#         GYRO_SCALE, DEADZONE, LPF_ALPHA, bias_gyro = 1., 0., 0., 0.
#         self.GYRO_SCALE = GYRO_SCALE
#         self.DEADZONE = DEADZONE
#         self.LPF_ALPHA = LPF_ALPHA
#         self.bias_gyro = bias_gyro
#
#         # 第一次滤波
#         self.omega_filtered = np.zeros(3)
#         self.omega_filtered_2nd = np.zeros(3)
#
#     def calibrate_imu(self):
#         imu = self.imu
#         state = self.state
#         AXIS_MAP = self.AXIS_MAP
#         RAW2RAD = self.RAW2RAD
#         RAW2ACC = self.RAW2ACC
#
#         calib_samples = []
#         print(f"[RawIMUHandler] calibrating...")
#
#         while len(calib_samples) < 800:
#             r, _, _ = select.select([imu.fd], [], [], 0)
#             if r:
#                 try:
#                     for evt in imu.read():
#                         if evt.type == ecodes.EV_ABS and evt.code in AXIS_MAP:
#                             state[AXIS_MAP[evt.code]] = evt.value
#                             if all(k in state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
#                                 omega_now = np.array([state['gx'], state['gy'], state['gz']]) * RAW2RAD
#                                 accel_now = np.array([state['ax'], state['ay'], state['az']]) * RAW2ACC
#                                 calib_samples.append((omega_now, accel_now))
#                 except BlockingIOError:
#                     pass
#
#         # 拆分样本
#         gyro_samples = np.array([o for o, a in calib_samples])
#         accel_samples = np.array([a for o, a in calib_samples])
#
#         # 陀螺仪校准
#         bias_gyro = np.mean(gyro_samples, axis=0)
#         gyro_std = np.std(gyro_samples, axis=0)
#         GYRO_SCALE = 0.03
#         DEADZONE_XYZ = 0.03 * gyro_std * GYRO_SCALE
#         LPF_ALPHA = np.clip(np.linalg.norm(gyro_std) * 20, 0.01, 0.1)
#
#         # 加速度计 bias 校准
#         bias_acc = np.mean(accel_samples, axis=0)
#         print(f"[RawIMUHandler][校准完成]")
#         print(f"  GYRO: bias={bias_gyro}, deadzone={DEADZONE_XYZ}, alpha={LPF_ALPHA:.5f}")
#         print(f"  ACC : bias={bias_acc}")
#
#         # 存下来
#         self.GYRO_SCALE = GYRO_SCALE
#         self.DEADZONE = DEADZONE_XYZ
#         self.LPF_ALPHA = LPF_ALPHA
#         self.bias_gyro = bias_gyro
#         self.bias_acc = bias_acc
#
#         return GYRO_SCALE, DEADZONE_XYZ, LPF_ALPHA, bias_gyro, bias_acc
#
#     def capture_imu_pose(self):
#         imu = self.imu
#         madgwick = self.madgwick
#         AXIS_MAP = self.AXIS_MAP
#         state = self.state
#         RAW2RAD = self.RAW2RAD
#         RAW2ACC = self.RAW2ACC
#
#         GYRO_SCALE = self.GYRO_SCALE
#         LPF_ALPHA = self.LPF_ALPHA
#         DEADZONE_XYZ = self.DEADZONE  # 注意现在是数组了
#         bias_gyro = self.bias_gyro
#
#         q_pose_old = self.q_pose.copy()
#         q_pose_new = self.q_pose.copy()
#
#         r, _, _ = select.select([imu.fd], [], [], 0)
#         if r:
#             try:
#                 for evt in imu.read():
#                     if evt.type == ecodes.EV_ABS and evt.code in AXIS_MAP:
#                         state[AXIS_MAP[evt.code]] = evt.value
#                         ts = evt.timestamp()
#                         t_now = ts if isinstance(ts, float) else ts[0] + ts[1] * 1e-6
#
#                         if all(k in state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
#                             omega_raw = np.array([state['gx'], state['gy'], state['gz']]) * RAW2RAD
#                             accel_raw = np.array([state['ax'], state['ay'], state['az']]) * RAW2ACC
#                             self.state_converted = self.cvt_imu_raw_data(state)
#
#                             omega = omega_raw
#
#                             # omega = (omega_raw - bias_gyro) * GYRO_SCALE
#
#                             # # 第一次滤波
#                             # self.omega_filtered = (1.0 - LPF_ALPHA) * self.omega_filtered + LPF_ALPHA * omega
#                             #
#                             # # 第二次滤波
#                             # if not hasattr(self, 'omega_filtered_2nd'):
#                             #     self.omega_filtered_2nd = np.zeros(3)
#                             # self.omega_filtered_2nd = ((1.0 - LPF_ALPHA) * self.omega_filtered_2nd +
#                             #                            LPF_ALPHA * self.omega_filtered)
#                             #
#                             # # 分别判断每个轴是否小于对应 deadzone
#                             # mask = np.abs(self.omega_filtered_2nd) < DEADZONE_XYZ
#                             # self.omega_filtered_2nd = self.omega_filtered_2nd * (~mask)
#
#                             if self.t_prev is not None:
#                                 dt = t_now - self.t_prev
#                                 madgwick.sampleperiod = dt
#                                 # q_pose_new = madgwick.updateIMU(copy.deepcopy(self.q_pose),
#                                 #                                 gyr=self.omega_filtered_2nd,
#                                 #                                 acc=accel_raw)
#                                 q_pose_new = omega
#                                 # q_rel = quat.qmult(quat.qinverse(q_pose_old), q_pose_new)
#                                 # rpy_rel = IMUControl.quat2euler(q_rel)
#                                 # self.rpy_rel = rpy_rel
#
#                             self.t_prev = t_now
#             except BlockingIOError:
#                 print("Error: BlockingIOError")
#             except Exception as e:
#                 print("Error: ", e)
#
#         self.q_pose = q_pose_new
#         return self.q_pose, self.rpy_rel
#
#     @staticmethod
#     def find_imu(name_keyword='Pro Controller (IMU)'):
#         for path in list_devices():
#             dev = InputDevice(path)
#             if name_keyword in dev.name:
#                 return dev
#         raise RuntimeError('IMU 未找到')
#
#     def cvt_imu_raw_data(self, raw_dict):
#         return {
#             'gx': ((raw_dict['gx'] >> 13) * self.GYRO_SCALE) * self.GYRO_TO_RAD,  # rad/s
#             'gy': ((raw_dict['gy'] >> 13) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
#             'gz': ((raw_dict['gz'] >> 13) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
#             'ax': raw_dict['ax'] * self.ACC_SCALE,  # m/s²
#             'ay': raw_dict['ay'] * self.ACC_SCALE,
#             'az': raw_dict['az'] * self.ACC_SCALE,
#         }


class ESKFIMU:
    def __init__(self, dt=0.005):
        self.dt = dt
        self.q = R.from_quat([0, 0, 0, 1])  # 四元数: x, y, z, w
        self.gyro_bias = np.zeros(3)
        self.P = np.eye(6) * 0.01
        self.Q = np.diag([1e-4]*3 + [1e-6]*3)
        self.R = np.eye(3) * 0.02  # 更高信任加速度观测，提升响应速度
        self.g = np.array([0.0, 0.0, -9.8])  # 重力方向（z向上）

    def predict(self, gyro):
        omega = gyro - self.gyro_bias
        delta_theta = omega * self.dt
        dq = R.from_rotvec(delta_theta)
        self.q = self.q * dq  # 四元数更新

        F = np.eye(6)
        F[0:3, 0:3] -= self.skew(omega) * self.dt
        F[0:3, 3:6] = -np.eye(3) * self.dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, acc):
        acc_norm = acc / np.linalg.norm(acc)
        acc_mag = np.linalg.norm(acc)

        # 仅在接近重力时才进行更新，滤除运动中的线性加速度干扰
        if np.abs(acc_mag - 9.8) > 0.5:
            return  # 跳过更新，等待静止状态

        g_b = self.q.inv().apply(self.g)
        g_b_norm = g_b / np.linalg.norm(g_b)
        y = acc_norm - g_b_norm

        H = np.zeros((3, 6))
        H[:, 0:3] = self.skew(g_b_norm)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = K @ y

        delta_theta = delta_x[0:3]
        delta_bias = delta_x[3:6]
        angle = np.linalg.norm(delta_theta)

        if np.isfinite(angle) and angle > 1e-8:
            dq = R.from_rotvec(delta_theta)
            self.q = dq * self.q

        self.gyro_bias += delta_bias
        self.P = (np.eye(6) - K @ H) @ self.P

    def step(self, acc, gyro):
        self.predict(gyro)
        self.update(acc)

    def get_euler(self):
        return self.q.as_euler('xyz', degrees=True)

    def skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


class MadgwickIMU:
    def __init__(self, dt=0.005, beta=0.1):
        self.dt = dt
        self.filter = Madgwick(sampleperiod=dt, beta=beta)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z 注意顺序！

    def step(self, acc, gyro):
        """
        Madgwick滤波器更新一步
        输入:
            acc: np.array([ax, ay, az]) 加速度, 单位 m/s²
            gyro: np.array([gx, gy, gz]) 角速度, 单位 rad/s
        """
        self.q = self.filter.updateIMU(self.q, gyr=gyro, acc=acc)

    def get_euler(self):
        """
        返回当前姿态欧拉角，单位是度，使用'sxyz'轴顺序
        """
        from transforms3d.euler import quat2euler
        # transforms3d要求四元数是 [w, x, y, z]
        roll, pitch, yaw = quat2euler(self.q, axes='sxyz')  # 返回的是弧度
        return np.degrees([roll, pitch, yaw])

    def predict(self, gyro):
        pass

    def update(self, acc):
        pass


def is_static(acc, acc_thresh=0.3, g=9.8):
    return np.abs(np.linalg.norm(acc) - g) < acc_thresh

# 正式 RawIMUHandler
class RawIMUHandlerV1:
    def __init__(self, device_name="Pro Controller (IMU)"):
        self.device_name = device_name
        self.imu = self.find_imu(device_name)
        print(f'[RawIMUHandler] Using {self.imu.path} ({self.imu.name})')

        self.AXIS_MAP = {
            ecodes.ABS_RX: 'gx', ecodes.ABS_RY: 'gy', ecodes.ABS_RZ: 'gz',
            ecodes.ABS_X: 'ax', ecodes.ABS_Y: 'ay', ecodes.ABS_Z: 'az',
        }

        self.GYRO_SCALE = (1 / 1000.) * 2000.0 / 32767.0
        self.GYRO_TO_RAD = np.pi / 180
        self.ACC_SCALE = 9.8 * 8.0 / 32767

        self.state = {'ax': 0, 'ay': 0, 'az': 0, 'gx': 0, 'gy': 0, 'gz': 0}
        self.state_converted = copy.deepcopy(self.state)
        self.t_prev = None

        self.bias_gyro = np.zeros(3)
        self.bias_acc = np.zeros(3)

        self.madgwick_imu = MadgwickIMU(dt=0.005, beta=0.01)
        self.q_pose = np.array([1.0, 0.0, 0.0, 0.0])  # 当前姿态
        self.q_calib = np.array([1.0, 0.0, 0.0, 0.0])  # 【改】校准基准姿态
        self.rpy_rel = np.array([0.0, 0.0, 0.0])

        self.calibrate_imu()

    def calibrate_imu(self):
        calib_samples = []
        print(f"[RawIMUHandler] Calibrating... Hold still...")
        # [校准完成] GYRO bias: [-0.00699018 -0.00165829 -0.00279665], ACC bias: [-1.76132121  0.03036882  8.56452744]
        # [校准完成] GYRO bias: [-0.00701541 -0.00152623 -0.00282055], ACC bias: [-1.79924973  0.03117465  8.86040988]
        # [校准完成] GYRO bias: [-0.00696372 -0.00159211 -0.00284389], ACC bias: [-1.79217931  0.03139588  9.34168081]

        import time
        start = time.time()
        while len(calib_samples) < 10000:
            r, _, _ = select.select([self.imu.fd], [], [], 0)
            if r:
                for evt in self.imu.read():
                    if evt.type == ecodes.EV_ABS and evt.code in self.AXIS_MAP:
                        self.state[self.AXIS_MAP[evt.code]] = evt.value
                        if all(k in self.state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                            self.state_converted = self.cvt_imu_raw_data(self.state)
                            omega = np.array([
                                self.state_converted['gx'],
                                self.state_converted['gy'],
                                self.state_converted['gz']
                            ])
                            accel = np.array([
                                self.state_converted['ax'],
                                self.state_converted['ay'],
                                self.state_converted['az']
                            ])
                            calib_samples.append((omega, accel))
        print(1000.0 * (time.time() - start))

        gyros = np.array([o for o, a in calib_samples])
        accs = np.array([a for o, a in calib_samples])
        self.bias_gyro = np.mean(gyros, axis=0)
        self.bias_acc = np.mean(accs, axis=0)

        print(f"[校准完成] GYRO bias: {self.bias_gyro}, ACC bias: {self.bias_acc}")

        # 【改】保存校准时的初始姿态
        self.q_calib = copy.deepcopy(self.madgwick_imu.q)

    def capture_imu_pose(self):
        r, _, _ = select.select([self.imu.fd], [], [], 0)
        if r:
            for evt in self.imu.read():
                if evt.type == ecodes.EV_ABS and evt.code in self.AXIS_MAP:
                    self.state[self.AXIS_MAP[evt.code]] = evt.value
                    ts = evt.timestamp()
                    t_now = ts if isinstance(ts, float) else ts[0] + ts[1] * 1e-6

                    if all(k in self.state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                        self.state_converted = self.cvt_imu_raw_data(self.state)
                        gyro = np.array([
                            self.state_converted['gx'],
                            self.state_converted['gy'],
                            self.state_converted['gz']
                        ])
                        acc = np.array([
                            self.state_converted['ax'],
                            self.state_converted['ay'],
                            self.state_converted['az']
                        ])

                        gyro -= self.bias_gyro
                        acc -= self.bias_acc

                        if self.t_prev is not None:
                            dt = t_now - self.t_prev
                            self.madgwick_imu.dt = dt
                            self.madgwick_imu.filter.sampleperiod = dt

                            self.madgwick_imu.step(acc, gyro)

                            self.q_pose = self.madgwick_imu.q

                            # 【改】计算 "当前姿态 相对于 校准姿态" 的四元数
                            q_rel = qmult(qinverse(self.q_calib), self.q_pose)

                            # 【改】转成欧拉角 (sxyz)
                            roll, pitch, yaw = quat2euler(q_rel, axes='sxyz')
                            self.rpy_rel = np.degrees([roll, pitch, yaw])

                            print(f'[DEBUG] dt={dt * 1000.:.2f}ms')

                        self.t_prev = t_now
        return self.q_pose, self.rpy_rel

    def cvt_imu_raw_data(self, raw_dict):
        """ No need to right-move for hid-nintendo event """
        return {
            'gx': ((raw_dict['gx']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'gy': ((raw_dict['gy']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'gz': ((raw_dict['gz']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'ax': raw_dict['ax'] * self.ACC_SCALE,
            'ay': raw_dict['ay'] * self.ACC_SCALE,
            'az': raw_dict['az'] * self.ACC_SCALE,
        }

    @staticmethod
    def find_imu(name_keyword='Pro Controller (IMU)'):
        for path in list_devices():
            dev = InputDevice(path)
            if name_keyword in dev.name:
                return dev
        raise RuntimeError('IMU 未找到')


class RawIMUHandler:
    def __init__(self, device_name="Pro Controller (IMU)"):
        self.device_name = device_name
        self.imu = self.find_imu(device_name)
        print(f'[RawIMUHandler] Using {self.imu.path} ({self.imu.name})')

        self.AXIS_MAP = {
            ecodes.ABS_RX: 'gx', ecodes.ABS_RY: 'gy', ecodes.ABS_RZ: 'gz',
            ecodes.ABS_X: 'ax', ecodes.ABS_Y: 'ay', ecodes.ABS_Z: 'az',
        }

        self.GYRO_SCALE = (1 / 1000.) * 2000.0 / 32767.0
        self.GYRO_TO_RAD = np.pi / 180
        self.ACC_SCALE = 9.8 * 8.0 / 32767

        self.state = {'ax': 0, 'ay': 0, 'az': 0, 'gx': 0, 'gy': 0, 'gz': 0}
        self.state_converted = copy.deepcopy(self.state)
        self.t_prev = None

        self.bias_gyro = np.zeros(3)
        self.bias_acc = np.zeros(3)

        self.q_pose = np.array([1.0, 0.0, 0.0, 0.0])  # 初始四元数 (w, x, y, z)
        self.rpy_rel = np.array([0.0, 0.0, 0.0])  # 相对角度（度）

        self.calibrate_imu()

    def calibrate_imu(self):
        calib_samples = []
        print(f"[RawIMUHandler] Calibrating... Please keep still...")

        while len(calib_samples) < 3000:
            r, _, _ = select.select([self.imu.fd], [], [], 0)
            if r:
                for evt in self.imu.read():
                    if evt.type == ecodes.EV_ABS and evt.code in self.AXIS_MAP:
                        self.state[self.AXIS_MAP[evt.code]] = evt.value
                        if all(k in self.state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                            self.state_converted = self.cvt_imu_raw_data(self.state)
                            omega = np.array([self.state_converted['gx'], self.state_converted['gy'], self.state_converted['gz']])
                            accel = np.array([self.state_converted['ax'], self.state_converted['ay'], self.state_converted['az']])
                            calib_samples.append((omega, accel))

        gyros = np.array([o for o, a in calib_samples])
        accs = np.array([a for o, a in calib_samples])

        self.bias_gyro = np.mean(gyros, axis=0)
        self.bias_acc = np.mean(accs, axis=0)

        print(f"[校准完成] GYRO bias: {self.bias_gyro}, ACC bias: {self.bias_acc}")

    def capture_imu_pose(self):
        r, _, _ = select.select([self.imu.fd], [], [], 0)
        if r:
            for evt in self.imu.read():
                if evt.type == ecodes.EV_ABS and evt.code in self.AXIS_MAP:
                    self.state[self.AXIS_MAP[evt.code]] = evt.value
                    ts = evt.timestamp()
                    t_now = ts if isinstance(ts, float) else ts[0] + ts[1] * 1e-6

                    if all(k in self.state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                        self.state_converted = self.cvt_imu_raw_data(self.state)
                        gyro = np.array([self.state_converted['gx'], self.state_converted['gy'], self.state_converted['gz']])
                        acc = np.array([self.state_converted['ax'], self.state_converted['ay'], self.state_converted['az']])

                        gyro -= self.bias_gyro
                        acc -= self.bias_acc

                        if self.t_prev is not None:
                            dt = t_now - self.t_prev

                            # 积分角速度
                            delta_theta = gyro * dt  # rad
                            angle = np.linalg.norm(delta_theta)

                            if np.isfinite(angle) and angle > 1e-8:
                                axis = delta_theta / angle
                                dq = self.axis_angle_to_quat(axis, angle)
                                self.q_pose = qmult(self.q_pose, dq)
                                self.q_pose /= np.linalg.norm(self.q_pose)


                            self.rpy_rel = self.quat_to_euler_continuous(self.q_pose)

                        self.t_prev = t_now

        return self.q_pose, self.rpy_rel

    def axis_angle_to_quat(self, axis, angle):
        w = np.cos(angle / 2)
        x, y, z = axis * np.sin(angle / 2)
        return np.array([w, x, y, z])

    def quat_to_euler_continuous(self, q):
        R = quat2mat(q)  # 四元数转成旋转矩阵
        euler = mat2euler(R, axes='sxyz')  # 从矩阵解出欧拉角
        return np.degrees(euler)  # 转成角度

    def cvt_imu_raw_data(self, raw_dict):
        return {
            'gx': ((raw_dict['gx']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'gy': ((raw_dict['gy']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'gz': ((raw_dict['gz']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'ax': raw_dict['ax'] * self.ACC_SCALE,
            'ay': raw_dict['ay'] * self.ACC_SCALE,
            'az': raw_dict['az'] * self.ACC_SCALE,
        }

    @staticmethod
    def find_imu(name_keyword='Pro Controller (IMU)'):
        for path in list_devices():
            dev = InputDevice(path)
            if name_keyword in dev.name:
                return dev
        raise RuntimeError('IMU 未找到')


from collections import deque
class RawIMUHandlerIncremental:
    def __init__(self, device_name="Pro Controller (IMU)"):
        self.device_name = device_name
        self.imu = self.find_imu(device_name)
        print(f'[RawIMUHandler] Using {self.imu.path} ({self.imu.name})')

        self.AXIS_MAP = {
            ecodes.ABS_RX: 'gy', ecodes.ABS_RY: 'gz', ecodes.ABS_RZ: 'gx',
            ecodes.ABS_X: 'ay', ecodes.ABS_Y: 'az', ecodes.ABS_Z: 'ax',
        }

        self.GYRO_SCALE = (1 / 1000.) * 2000.0 / 32767.0
        self.GYRO_TO_RAD = np.pi / 180
        self.ACC_SCALE = 9.8 * 8.0 / 32767

        self.ACC_SCALE = 9.80665 / 8192  # m/s²
        self.GYRO_SCALE = 1 / 1024  # rad/s

        self.state = {'ax': 0, 'ay': 0, 'az': 0, 'gx': 0, 'gy': 0, 'gz': 0}
        self.state_converted = copy.deepcopy(self.state)
        self.t_prev = None

        self.bias_gyro = np.zeros(3)
        self.bias_acc = np.zeros(3)

        self.q_last = np.array([1.0, 0.0, 0.0, 0.0])  # 上一帧
        self.q_cum = np.array([1.0, 0.0, 0.0, 0.0])   # 累积
        self.rpy_rel = np.zeros(3)
        self.rpy_cum = np.zeros(3)

        self.calibrate_imu()

    def calibrate_imu(self):
        calib_samples = []
        print(f"[RawIMUHandler] Calibrating... Please keep still...")

        while len(calib_samples) < 3000:
            r, _, _ = select.select([self.imu.fd], [], [], 0)
            if r:
                for evt in self.imu.read():
                    if evt.type == ecodes.EV_ABS and evt.code in self.AXIS_MAP:
                        self.state[self.AXIS_MAP[evt.code]] = evt.value
                        if all(k in self.state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                            self.state_converted = self.cvt_imu_raw_data(self.state)
                            omega = np.array(
                                [self.state_converted['gx'], self.state_converted['gy'], self.state_converted['gz']])
                            accel = np.array(
                                [self.state_converted['ax'], self.state_converted['ay'], self.state_converted['az']])
                            calib_samples.append((omega, accel))

        gyros = np.array([o for o, a in calib_samples])
        accs = np.array([a for o, a in calib_samples])

        self.bias_gyro = np.mean(gyros, axis=0)
        self.bias_acc = np.mean(accs, axis=0)

        print(f"[校准完成] GYRO bias: {self.bias_gyro}, ACC bias: {self.bias_acc}")

    def capture_imu_pose(self):
        r, _, _ = select.select([self.imu.fd], [], [], 0)
        if r:
            for evt in self.imu.read():
                if evt.type == ecodes.EV_ABS and evt.code in self.AXIS_MAP:
                    self.state[self.AXIS_MAP[evt.code]] = evt.value
                    ts = evt.timestamp()
                    t_now = ts if isinstance(ts, float) else ts[0] + ts[1] * 1e-6

                    if all(k in self.state for k in ('gx', 'gy', 'gz', 'ax', 'ay', 'az')):
                        self.state_converted = self.cvt_imu_raw_data(self.state)
                        gyro = np.array([
                            self.state_converted['gx'],
                            self.state_converted['gy'],
                            self.state_converted['gz']
                        ])
                        acc = np.array([
                            self.state_converted['ax'],
                            self.state_converted['ay'],
                            self.state_converted['az']
                        ])

                        gyro -= self.bias_gyro
                        acc -= self.bias_acc

                        if self.t_prev is not None:
                            dt = t_now - self.t_prev
                            omega = gyro * dt  # rad

                            angle = np.linalg.norm(omega)
                            if angle > 1e-8:
                                axis = omega / angle
                                dq = axangle2quat(axis, angle)
                            else:
                                dq = np.array([1.0, 0.0, 0.0, 0.0])

                            # 这次小变化
                            self.q_rel = dq
                            self.rpy_rel = np.degrees(quat2euler(dq, axes='sxyz'))

                            # 累加
                            self.q_cum = qmult(self.q_cum, dq)
                            self.rpy_cum = np.degrees(quat2euler(self.q_cum, axes='sxyz'))

                        self.t_prev = t_now

        return self.rpy_rel, self.rpy_cum

    def cvt_imu_raw_data(self, raw_dict):
        return {
            'gx': ((raw_dict['gx']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'gy': ((raw_dict['gy']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'gz': ((raw_dict['gz']) * self.GYRO_SCALE) * self.GYRO_TO_RAD,
            'ax': raw_dict['ax'] * self.ACC_SCALE,
            'ay': raw_dict['ay'] * self.ACC_SCALE,
            'az': raw_dict['az'] * self.ACC_SCALE,
        }

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
