import time
import os
from typing import List
import numpy as np
from PIL import Image

from robokit.service.service_connector import ServiceConnector
from robokit.network.robot_client import RobotClient
from robokit.data.realsense_handler import RealsenseHandler
from robokit.data.data_handler import DataHandler


class RealWorldEvaluator:
    """ Run on Robot arm local network. """
    def __init__(self,
                 gpu_service_connector: ServiceConnector,
                 robot: RobotClient,
                 run_loops: int = 100,
                 img_hw: tuple = (256, 256),
                 fps: int = 5,
                 enable_auto_ae_wb: bool = True,
                 ):
        self.connector = gpu_service_connector
        self.robot = robot
        self.run_loops = run_loops
        self.img_hw = img_hw
        self.start_time = time.time()

        self.fps = fps
        self.camera = RealsenseHandler(frame_rate=60)
        self.camera.set_ae_wb_auto(enable_auto_ae_wb)
        if not enable_auto_ae_wb:
            self.camera.set_ae_wb(exposure=100)

        # Dynamic variables
        self.step_cnt = 0
        self.g = 0.
        self.frame_buffer = {
            'primary_rgb': [],
            'gripper_rgb': [],
            'joint_state': [],
        }  # some policies require 2 or more observation frames

        # Record
        self.log_time_delays = []

    def time_tick(self):
        self.start_time = time.time()

    def time_tok(self):
        time_delay = time.time() - self.start_time
        print(f"[DebugEvaluator] time cost: {time_delay * 1000.:.1f} ms")
        self.log_time_delays.append(time_delay)

    def show_time_delays(self):
        delays = np.array(self.log_time_delays) * 1000.  # second -> ms
        fps_lists = (1000. / (delays + 1e-6)).astype(np.int32)
        print(f"[DebugEvaluator] time delay: min={np.min(delays):.1f} ms, max={np.max(delays):.1f} ms, "
              f"avg={np.average(delays):.1f} ms; "
              f"min_fps={np.min(fps_lists)} FPS, max_fps={np.max(fps_lists)} FPS, "
              f"avg_fps={np.average(fps_lists)} FPS.")

    def run(self) -> float:
        last_game_time = time.time()  # for FPS limitation

        cur_task_text = "pick up the banana"
        self.connector.reset(cur_task_text)
        self.reset()

        while True:
            self.time_tick()

            current_time = time.time()
            if current_time - last_game_time >= 1. / self.fps:  # Robotic Arm FPS constraint
                last_game_time = current_time

                # 1. Get current scene observation
                frame_obs = self.capture_env_observation()

                cur_primary_rgb = frame_obs['primary_rgb']
                cur_gripper_rgb = frame_obs['gripper_rgb']
                # cur_joint_state = frame_obs['robot_obs'].tolist()[7:13]  # Real joint_state
                cur_joint_state = frame_obs['robot_obs'].tolist()[:6]  # TCP pos as joint_state

                if len(self.frame_buffer['primary_rgb']) == 0:
                    self.frame_buffer['primary_rgb'].append(cur_primary_rgb)
                    self.frame_buffer['gripper_rgb'].append(cur_gripper_rgb)
                    self.frame_buffer['joint_state'].append(cur_joint_state)
                    cur_primary_rgb = np.stack([np.zeros_like(cur_primary_rgb), cur_primary_rgb])
                    cur_joint_state = [[0.] * len(cur_joint_state), cur_joint_state]
                elif len(self.frame_buffer['primary_rgb']) == 1:
                    self.frame_buffer['primary_rgb'].append(cur_primary_rgb)
                    self.frame_buffer['gripper_rgb'].append(cur_gripper_rgb)
                    self.frame_buffer['joint_state'].append(cur_joint_state)
                    cur_primary_rgb = np.stack(self.frame_buffer['primary_rgb'])
                    cur_joint_state = self.frame_buffer['joint_state']
                else:
                    assert len(self.frame_buffer['primary_rgb']) == 2
                    cur_primary_rgb = np.stack(self.frame_buffer['primary_rgb'])
                    cur_gripper_rgb = np.stack(self.frame_buffer['gripper_rgb'])
                    cur_joint_state = list(self.frame_buffer['joint_state'])
                    self.frame_buffer['primary_rgb'].pop(0)
                    self.frame_buffer['gripper_rgb'].pop(0)
                    self.frame_buffer['joint_state'].pop(0)

                # 2. Send observation to GPU model, to get action
                print(cur_primary_rgb[-1].shape)
                action = self.connector.step(
                    primary_rgb=cur_primary_rgb,  # (T,H,W,C) to List[str]
                    gripper_rgb=cur_gripper_rgb,
                    task_description=cur_task_text,
                    joint_state=cur_joint_state,  # [[0.] * 6] * T
                )
                assert action.shape[1] == 7
                print(self.step_cnt, action, cur_joint_state)

                # 4. Conduct action in real-world environment
                self.step(action[0, :])
                self.on_gripper_move()
                # Robot arm conducting delay
                self.time_tok()
                self.step_cnt += 1

                if self.step_cnt >= self.run_loops:
                    break  # Finished
            else:
                pass

            time.sleep(1. / 120)  # Env FPS, like pygame data collection

        self.show_time_delays()
        return 0.

    def capture_env_observation(self) -> dict:
        camera_data = self.camera.capture_frames()
        robot_data = self.robot.get_current_frame_info()

        frame_robot_obs = np.array([
            robot_data['tcp_xyz_wrt_base']['x'],
            robot_data['tcp_xyz_wrt_base']['y'],
            robot_data['tcp_xyz_wrt_base']['z'],
            robot_data['tcp_ori_wrt_base'][0],
            robot_data['tcp_ori_wrt_base'][1],
            robot_data['tcp_ori_wrt_base'][2],
            robot_data['gripper_width'],
            robot_data['joint_states'][0],
            robot_data['joint_states'][1],
            robot_data['joint_states'][2],
            robot_data['joint_states'][3],
            robot_data['joint_states'][4],
            robot_data['joint_states'][5],
            robot_data['gripper_moving_to'],
        ])  # shape:(14,)

        env_obs_dict = {
            "primary_rgb": camera_data['color2'],
            "gripper_rgb": camera_data['color1'],
            "primary_depth": camera_data['depth2'],
            "gripper_depth": camera_data['depth1'],
            "robot_obs": frame_robot_obs,
        }
        return env_obs_dict

    def step(self, action: np.ndarray) -> None:
        """ Send action to Robotic Arm """
        linear_xyz = {'x': action[0], 'y': action[1], 'z': action[2]}
        angular_xyz = {'x': action[3], 'y': action[4], 'z': action[5]}
        self.g = float(action[6])
        if abs(action[3]) + abs(action[4]) + abs(action[5]) > abs(action[0]) + abs(action[1]) + abs(action[2]):
            self.robot.ang_jog_pub(angular_xyz)
            print("angular")
        else:
            self.robot.linear_jog_pub(linear_xyz)
            print("linear")

    def on_gripper_move(self, g: float = None) -> None:
        self.g = g if g is not None else self.g
        self.robot.gripper_set_pub(self.g)

    def reset(self):
        # Sanity check for gripper
        self.robot.gripper_set_pub(1)
        time.sleep(0.5)
        self.robot.gripper_set_pub(0)
        time.sleep(0.5)

        # Go back home
        self.robot.joint_back_home()