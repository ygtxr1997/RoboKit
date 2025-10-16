import numpy as np
import copy

from robokit.network.vr_handler import QuestHandler
from robokit.debug_utils.images import DynamicDataDrawer


class QuestVisualizer(DynamicDataDrawer):
    def __init__(self, max_points: int = 200):
        self.raw_quest = QuestHandler()
        super().__init__(self.raw_quest,
                         data_keys=[
                             ['X', 'Y', 'Z'],
                             ['dX', 'dY', 'dZ'],
                             ['roll', 'pitch', 'yaw'],
                             ['roll_rel', 'pitch_rel', 'yaw_rel'],
                         ],
                         y_minmax_values=[
                             [-2, 2],
                             [-1, 1],
                             [-200, 200],
                             [-24, 24]
                         ],
                         max_points=max_points)

    def get_new_data(self) -> dict:
        latest_data = self.raw_quest.get_latest_euler()
        xyz_now = latest_data['now_xyz']
        rpy_now = latest_data['now_euler']
        xyz_rel = latest_data['xyz']
        rpy_rel = latest_data['euler']

        # l_transform = transform_data['l']  # (4x4, array)
        # r_transform = transform_data['r']  # (4x4, array)
        # button = button_data

        # rpy_now = euler_data['euler'][:3]
        # rpy_rel = euler_data['euler'][3:6]
        # imu_data = copy.deepcopy(self.raw_quest.get_latest_imu())

        vis_data = {}
        vis_data['X'], vis_data['Y'], vis_data['Z'] = [float(x) for x in xyz_now]
        vis_data['dX'], vis_data['dY'], vis_data['dZ'] = [float(x) for x in xyz_rel]
        vis_data['roll'], vis_data['pitch'], vis_data['yaw'] = [float(x) for x in rpy_now]
        vis_data['roll_rel'], vis_data['pitch_rel'], vis_data['yaw_rel'] = [float(x) for x in rpy_rel]
        # print('RPY:', result_rpy)
        return vis_data


visualizer = QuestVisualizer()

try:
    visualizer.run()
except KeyboardInterrupt:
    print("Exiting by KeyboardInterrupt")
finally:
    visualizer.raw_quest.stop()
    print("IMU closed")

