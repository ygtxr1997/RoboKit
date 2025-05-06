import copy
import multiprocessing as mp
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


def get_gamepad_hyperparams(gamepad_name: str):
    gamepad_name = gamepad_name.lower()
    assert gamepad_name in ['ps5_dualsense', 'ns1_pro'], "Invalid gamepad name"
    if gamepad_name == 'ps5_dualsense':
        return {
            'evdev_name': 'DualSense Wireless Controller Motion Sensors',
            'acc_scale': 9.80665 / 8192,  # m/s²
            'gyro_scale': 1 / 1024,  # rad/s
            'axis_map': {
                ecodes.ABS_RX: 'gy', ecodes.ABS_RY: 'gz', ecodes.ABS_RZ: 'gx',
                ecodes.ABS_X: 'ay', ecodes.ABS_Y: 'az', ecodes.ABS_Z: 'ax',
            }
        }
    elif gamepad_name == 'ns1_pro':
        return {
            'evdev_name': 'Pro Controller (IMU)',
            'acc_scale': 9.80665 * 8.0 / 32767,  # m/s²
            'gyro_scale': (1 / 1000.) * 2000.0 / 32767.0,  # rad/s
            'axis_map': {
                ecodes.ABS_RX: 'gx', ecodes.ABS_RY: 'gy', ecodes.ABS_RZ: 'gz',
                ecodes.ABS_X: 'ax', ecodes.ABS_Y: 'ay', ecodes.ABS_Z: 'az',
            }
        }
    else:
        raise ValueError("Unsupported gamepad name")


import numpy as np
from transforms3d.quaternions import qmult, qinverse
from transforms3d.euler import quat2euler


class ESKF6D:
    def __init__(self,
                 gyro_noise=1e-6,
                 gyro_bias_noise=1e-8,
                 acc_noise=1e-2):
        # 主状态
        self.q    = np.array([1.,0.,0.,0.])   # 姿态四元数
        self.b_g  = np.zeros(3)               # 陀螺偏置

        # 误差卡尔曼协方差 P (6×6)
        self.P    = np.eye(6) * 1e-4

        # 过程噪声 Q
        self.Q    = np.diag([gyro_noise]*3 + [gyro_bias_noise]*3)

        # 测量噪声 R (针对单位重力方向)
        self.R    = np.eye(3) * (acc_noise**2)

        # 参考重力方向（单位向量）
        self.g_ref = np.array([0., 0., 1.])

    @staticmethod
    def skew(v):
        return np.array([[    0, -v[2],  v[1]],
                         [ v[2],     0, -v[0]],
                         [-v[1],  v[0],     0]])

    @staticmethod
    def small_quat(dtheta):
        angle = np.linalg.norm(dtheta)
        if angle < 1e-8:
            return np.array([1.,0.,0.,0.])
        axis = dtheta / angle
        s = np.sin(angle/2)
        return np.array([np.cos(angle/2), axis[0]*s, axis[1]*s, axis[2]*s])

    def predict(self, gyro, dt):
        # 去偏
        omega = gyro - self.b_g

        # 四元数积分
        w,x,y,z = self.q
        Omega = np.array([
            [   0,    -omega[0], -omega[1], -omega[2]],
            [omega[0],     0,     omega[2], -omega[1]],
            [omega[1], -omega[2],     0,     omega[0]],
            [omega[2],  omega[1], -omega[0],     0    ]
        ])
        q_dot = 0.5 * Omega.dot(self.q)
        self.q = self.q + q_dot * dt
        self.q /= np.linalg.norm(self.q)

        # 构造 Jacobian F
        F = np.zeros((6,6))
        F[0:3,0:3] = -self.skew(omega)
        F[0:3,3:6] = -np.eye(3)

        # 协方差离散传播
        self.P = self.P + (F.dot(self.P) + self.P.dot(F.T) + self.Q) * dt

    def update(self, acc):
        # 归一化测量
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-6:
            return
        a = acc / acc_norm

        # 预测的重力方向在机体系：g_body = q ⊗ [0,g_ref] ⊗ q⁻¹，然后取向量部分
        q_conj = qinverse(self.q)
        g_body_full = qmult(qmult(self.q, np.hstack([0., self.g_ref])), q_conj)
        g_body = g_body_full[1:]

        # 创新
        y = a - g_body

        # 测量矩阵 H = [∂g_body/∂δθ , 0]
        H = np.zeros((3,6))
        H[:,0:3] = self.skew(g_body)

        # 卡尔曼增益
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        # 误差状态
        dx = K.dot(y)
        dtheta = dx[0:3]
        dbias  = dx[3:6]

        # 修正主状态
        dq = self.small_quat(dtheta)
        self.q  = qmult(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        self.b_g += dbias

        # 更新协方差
        I = np.eye(6)
        self.P = (I - K.dot(H)).dot(self.P)

    def get_rpy(self):
        return quat2euler(self.q, axes='sxyz')


class RawIMUHandler:
    def __init__(self, game_pad_name='ps5_dualsense'):

        gamepad_hyperparams = get_gamepad_hyperparams(game_pad_name)
        self.device_name = gamepad_hyperparams['evdev_name']
        self.ACC_SCALE = gamepad_hyperparams['acc_scale']
        self.GYRO_SCALE = gamepad_hyperparams['gyro_scale']
        self.GYRO_TO_RAD = np.pi / 180
        self.AXIS_MAP = gamepad_hyperparams['axis_map']

        self.imu = self.find_imu(self.device_name)
        print(f'[RawIMUHandler] Using {self.imu.path} ({self.imu.name})')

        self.state = {'ax': 0, 'ay': 0, 'az': 0, 'gx': 0, 'gy': 0, 'gz': 0}
        self.state_converted = copy.deepcopy(self.state)

        self.q_last = np.array([1.0, 0.0, 0.0, 0.0])  # last pose, output by get_latest_euler()
        self.q_now = np.array([1.0, 0.0, 0.0, 0.0])   # current pose
        self.q_real = np.array([1.0, 0.0, 0.0, 0.0])  # filtered current pose, but filter is not used for now
        self.rpy_rel = np.zeros(3)  # q_now <- q_last + rpy_rel
        self.rpy_now = np.zeros(3)  # 'sxyz' order of current pose ``q_now``
        self.rpy_real = np.zeros(3)  # 'sxyz' order of filtered current pose ``q_real``
        self.t_prev = None
        self.t_start = None

        # Multiprocessing related
        self._reset_evt = mp.Event()
        self._acquire_euler_evt = mp.Event()
        self._euler_arr = mp.Array('d', 6)  # 共享欧拉角（rpy_real, rpy_rel）
        self._quat_arr = mp.Array('d', 4)  # 共享四元数（q_real）
        self._imu_arr = mp.Array('d', 6)  # shared IMU data (6 doubles)
        self._lock = mp.Lock()
        self._running = mp.Value('b', True)  # 控制子进程运行状态
        self._process = mp.Process(target=self._imu_loop, args=(
            self._lock, self._running
        ))

        # Calibration
        self.bias_gyro = np.zeros(3)
        self.bias_acc = np.zeros(3)
        self.calibrate_imu()

        # Start the IMU loop in a separate subprocess
        self._process.start()

    def _init_state(self):
        # 原始积分相关
        self.q_last = np.array([1.0, 0.0, 0.0, 0.0])
        self.q_now = np.array([1.0, 0.0, 0.0, 0.0])
        self.rpy_rel = np.zeros(3)
        self.rpy_now = np.zeros(3)

        # ESKF 滤波后“真实”姿态
        self.q_real = np.array([1.0, 0.0, 0.0, 0.0])
        self.rpy_real = np.zeros(3)

        # 时间戳
        self.t_prev = None
        self.t_start = None

        # 清空共享内存
        with self._lock:
            for i in range(6):
                self._euler_arr[i] = 0.0
            for i in range(4):
                self._quat_arr[i] = 0.0
            for i in range(6):
                self._imu_arr[i] = 0.0

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
                        if self.t_start is None:
                            self.t_start = t_now
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
                            if angle > 1e-13:
                                axis = omega / angle
                                dq = axangle2quat(axis, angle)
                            else:
                                dq = np.array([1.0, 0.0, 0.0, 0.0])

                            # dq: 这次小变化, 1000Hz
                            d_rpy = np.degrees(quat2euler(dq, axes='sxyz'))

                            # 累加
                            self.q_now = qmult(self.q_now, dq)
                            self.rpy_now = np.degrees(quat2euler(self.q_now, axes='sxyz'))

                            # no filters for now
                            self.q_real = self.q_now
                            self.rpy_real = self.rpy_now

                        self.t_prev = t_now

        return self.rpy_now, self.rpy_rel

    def _imu_loop(self, lock, running):
        print("[RawIMUHandler] IMU loop started")
        while running.value:
            if self._reset_evt.is_set():
                self._reset_evt.clear()
                self._init_state()

            if self._acquire_euler_evt.is_set():
                # Calculate: rpy_rel
                # Set: q_last
                self._acquire_euler_evt.clear()
                dq = qmult(self.q_real, qinverse(self.q_last))
                self.rpy_rel = np.degrees(quat2euler(dq, axes='sxyz'))
                self.q_last = self.q_real

            self.capture_imu_pose()

            with lock:
                self._euler_arr[0] = self.rpy_real[0]
                self._euler_arr[1] = self.rpy_real[1]
                self._euler_arr[2] = self.rpy_real[2]
                self._euler_arr[3] = self.rpy_rel[0]
                self._euler_arr[4] = self.rpy_rel[1]
                self._euler_arr[5] = self.rpy_rel[2]

                self._quat_arr[0] = self.q_real[0]
                self._quat_arr[1] = self.q_real[1]
                self._quat_arr[2] = self.q_real[2]
                self._quat_arr[3] = self.q_real[3]

                self._imu_arr[0] = self.state_converted['ax']
                self._imu_arr[1] = self.state_converted['ay']
                self._imu_arr[2] = self.state_converted['az']
                self._imu_arr[3] = self.state_converted['gx']
                self._imu_arr[4] = self.state_converted['gy']
                self._imu_arr[5] = self.state_converted['gz']

    def reset_pose(self):
        self._reset_evt.set()
        print("[RawIMUHandler] Pose reset")

    def get_latest_euler(self):
        self._acquire_euler_evt.set()
        with self._lock:
            return {
                'euler': np.array(self._euler_arr[:]),
                'quat': np.array(self._quat_arr[:]),
            }

    def get_latest_imu(self):
        with self._lock:
            return {
                'ax': self._imu_arr[0],
                'ay': self._imu_arr[1],
                'az': self._imu_arr[2],
                'gx': self._imu_arr[3],
                'gy': self._imu_arr[4],
                'gz': self._imu_arr[5],
            }

    def stop(self):
        self._running.value = False
        self._process.join(timeout=1)

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
