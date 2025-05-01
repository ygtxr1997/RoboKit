import numpy as np
import copy

from robokit.network.imu_control import RawIMUHandler, ESKFIMU, RawIMUHandlerIncremental
from robokit.debug_utils.images import DynamicDataDrawer

"""
Useful commands:

cat /proc/bus/input/devices

evtest /dev/input/event18

sudo usermod -a -G input $USER
newgrp input      # 或者重新登录一


"""


class IMUVisualizer(DynamicDataDrawer):
    def __init__(self, max_points: int = 200):
        self.raw_imu = RawIMUHandlerIncremental()
        super().__init__(self.raw_imu,
                         data_keys=[['ax', 'ay', 'az'], ['gx', 'gy', 'gz'], ['roll', 'pitch', 'yaw']],
                         max_points=max_points)

    def get_new_data(self) -> dict:
        q_pose, rpy_rel = self.raw_imu.capture_imu_pose()
        new_data = copy.deepcopy(self.raw_imu.state_converted)

        # acc = np.array([new_data['ax'], new_data['ay'], new_data['az']])
        # gyro = np.array([new_data['gx'], new_data['gy'], new_data['gz']])
        # self.eskf.step(acc, gyro)

        # result_rpy = self.eskf.get_euler()

        vis_data = copy.deepcopy(new_data)
        rpy_rel = self.raw_imu.rpy_cum
        vis_data['roll'], vis_data['pitch'], vis_data['yaw'] = [float(x) for x in rpy_rel]
        # print('RPY:', result_rpy)
        return vis_data


imu_visualizer = IMUVisualizer()
imu_visualizer.run()
