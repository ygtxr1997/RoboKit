import numpy as np
import copy

from robokit.network.imu_control import RawIMUHandler
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
        self.raw_imu = RawIMUHandler()
        super().__init__(self.raw_imu,
                         data_keys=[
                             ['ax', 'ay', 'az'],
                             ['gx', 'gy', 'gz'],
                             ['roll', 'pitch', 'yaw'],
                             ['roll_rel', 'pitch_rel', 'yaw_rel'],
                         ],
                         max_points=max_points)

    def get_new_data(self) -> dict:
        euler_data = self.raw_imu.get_latest_euler()
        rpy_now = euler_data['euler'][:3]
        rpy_rel = euler_data['euler'][3:6]
        imu_data = copy.deepcopy(self.raw_imu.get_latest_imu())

        vis_data = copy.deepcopy(imu_data)
        vis_data['roll'], vis_data['pitch'], vis_data['yaw'] = [float(x) for x in rpy_now]
        vis_data['roll_rel'], vis_data['pitch_rel'], vis_data['yaw_rel'] = [float(x) for x in rpy_rel]
        # print('RPY:', result_rpy)
        return vis_data


imu_visualizer = IMUVisualizer()

try:
    imu_visualizer.run()
except KeyboardInterrupt:
    print("Exiting by KeyboardInterrupt")
finally:
    imu_visualizer.raw_imu.stop()
    print("IMU closed")

