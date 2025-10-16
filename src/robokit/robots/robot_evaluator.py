import time
import os
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image

from robokit.connects.service_connector import ServiceConnector
from robokit.robots.robot_client_inovo import RobotClient
from robokit.data_manager.realsense_handler import RealsenseHandler
from robokit.data_manager.data_handler import DataHandler, ImageAsVideoSaver, ActionAsVideoSaver
from robokit.debug_utils.images import concatenate_rgb_images


class RealWorldEvaluator:
    """ Run on Robot arm local robots. """
    def __init__(self,
                 gpu_service_connector: ServiceConnector,
                 robot: RobotClient,
                 cur_task_text: str,
                 run_loops: int = 100,
                 img_hw: tuple = (256, 256),
                 resize_hw: tuple = None,
                 fps: int = 5,
                 enable_auto_ae_wb: bool = True,
                 buffer_size: int = 1,
                 speed_scale: float = 0.75,
                 # Save as video
                 image_save_speedup: float = 2.0,
                 image_save_fn: str = "tmp_saved_video.mp4",
                 action_save_flag: bool = False,
                 ):
        self.connector = gpu_service_connector
        self.robot = robot
        self.cur_task_text = cur_task_text
        self.run_loops = run_loops
        self.img_hw = img_hw
        self.connector.resize_hw = resize_hw
        self.start_time = time.time()

        self.fps = fps
        self.camera = RealsenseHandler(
            frame_rate=60,
            img_height=img_hw[0],
            img_width=img_hw[1],
            use_depth=False
        )
        self.camera.set_ae_wb_auto(enable_auto_ae_wb)
        if not enable_auto_ae_wb:
            self.camera.set_ae_wb(exposure=50)

        self.speed_scale = speed_scale

        # Dynamic variables
        self.step_cnt = 0
        self.g = 0.
        self.frame_buffer = {
            'primary_rgb': [],
            'gripper_rgb': [],
            'joint_state': [],
        }  # some policies require 2 or more observation frames
        self.buffer_size = buffer_size

        # Record
        self.log_time_delays = []
        self.image_save_speedup = image_save_speedup
        self.image_save_fn = image_save_fn
        self.image_saver = ImageAsVideoSaver(
            buffer_size=run_loops,
            frame_rate=int(self.fps * self.image_save_speedup),
            height=img_hw[0],
            width=img_hw[1] * 2,
        )
        self.action_save_flag = action_save_flag
        self.action_save_fn = image_save_fn.replace('.mp4', '_action.mp4')
        self.action_saver = ActionAsVideoSaver(
            buffer_size=run_loops,
            frame_rate=int(self.fps * self.image_save_speedup),
            height=4.5,
            width=6,
        )

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

        # cur_task_text = "pick up the banana"
        # cur_task_text = "pick up the shovel and put it into the cup"
        cur_task_text = self.cur_task_text
        buffer_size = self.buffer_size
        self.connector.init_socket(cur_task_text)
        self.connector.send_reset(cur_task_text)
        self.reset()

        for _ in tqdm(range(200), desc="Skipping frames:"):
            self.capture_env_observation()

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

                # if len(self.frame_buffer['primary_rgb']) == 0:
                #     self.frame_buffer['primary_rgb'].append(cur_primary_rgb)
                #     self.frame_buffer['gripper_rgb'].append(cur_gripper_rgb)
                #     self.frame_buffer['joint_state'].append(cur_joint_state)
                #     cur_primary_rgb = np.stack([np.zeros_like(cur_primary_rgb), cur_primary_rgb])
                #     cur_gripper_rgb = np.stack([np.zeros_like(cur_gripper_rgb), cur_gripper_rgb])
                #     cur_joint_state = [[0.] * len(cur_joint_state), cur_joint_state]
                # elif len(self.frame_buffer['primary_rgb']) == 1:
                #     self.frame_buffer['primary_rgb'].append(cur_primary_rgb)
                #     self.frame_buffer['gripper_rgb'].append(cur_gripper_rgb)
                #     self.frame_buffer['joint_state'].append(cur_joint_state)
                #     cur_primary_rgb = np.stack(self.frame_buffer['primary_rgb'])
                #     cur_gripper_rgb = np.stack(self.frame_buffer['gripper_rgb'])
                #     cur_joint_state = self.frame_buffer['joint_state']
                # else:
                #     assert len(self.frame_buffer['primary_rgb']) == 2
                #     cur_primary_rgb = np.stack(self.frame_buffer['primary_rgb'])
                #     cur_gripper_rgb = np.stack(self.frame_buffer['gripper_rgb'])
                #     cur_joint_state = list(self.frame_buffer['joint_state'])
                #     self.frame_buffer['primary_rgb'].pop(0)
                #     self.frame_buffer['gripper_rgb'].pop(0)
                #     self.frame_buffer['joint_state'].pop(0)

                # --- 更新帧缓存 ---
                # 如果未满，直接 append；满了先 pop 再 append
                for key, val in zip(
                        ['primary_rgb', 'gripper_rgb', 'joint_state'],
                        [cur_primary_rgb, cur_gripper_rgb, cur_joint_state]
                ):
                    if len(self.frame_buffer[key]) >= buffer_size:
                        self.frame_buffer[key].pop(0)
                    self.frame_buffer[key].append(val)

                # --- 构造输出（长度一定为 buffer_size） ---
                def pad_to_buffer(data_list, pad_with):
                    padded = [pad_with for _ in range(buffer_size - len(data_list))] + data_list
                    return np.stack(padded) if isinstance(pad_with, np.ndarray) else padded


                cur_primary_rgb = pad_to_buffer(
                    self.frame_buffer['primary_rgb'],
                    np.zeros_like(cur_primary_rgb)
                )
                cur_gripper_rgb = pad_to_buffer(
                    self.frame_buffer['gripper_rgb'],
                    np.zeros_like(cur_gripper_rgb)
                )
                cur_joint_state = pad_to_buffer(
                    self.frame_buffer['joint_state'],
                    [0.] * len(cur_joint_state)
                    )

                # Save current image
                saved_image = concatenate_rgb_images(
                    self.frame_buffer['primary_rgb'][-1],
                    self.frame_buffer['gripper_rgb'][-1],
                )
                self.image_saver.add_image(saved_image)  # last is current

                # 2. Send observation to GPU model, to get action
                # print(cur_primary_rgb[-1].shape)
                action = self.connector.send_obs_and_get_action(
                    primary_rgb=cur_primary_rgb[None, :],  # (1,T,H,W,C) to List[str]
                    gripper_rgb=cur_gripper_rgb[None, :],
                    task_description=cur_task_text,
                    joint_state=np.array(cur_joint_state)[None, :],  # [[0.] * 6] * T -> (1,T,6)
                )
                assert action.shape[-1] == 7, f"action.shape[-1]={action.shape} is not (:,7)"
                print("[Info]", self.step_cnt,
                      self.format_array(action), "\n",
                      self.format_array(np.array(cur_joint_state)),
                      cur_task_text[:20],
                )

                ## Slow down when the height is too low
                # safe_min_z = 0.2
                # print(cur_joint_state[-1][2], safe_min_z)
                # if cur_joint_state[-1][2] <= safe_min_z:
                #     for h in range(action.shape[0]):
                #         if action[h, 2] < 0.:
                #             action[h, 2] = 0.7 * action[h, 2]

                # 4. Conduct action in real-world environment
                self.step(action_D=action[0, 0, :])
                self.on_gripper_move()
                # Robot arm conducting delay
                self.time_tok()
                self.step_cnt += 1

                # Save current action
                self.action_saver.add_action(action[0, 0, :])

                if self.step_cnt >= self.run_loops:
                    break  # Finished
            else:
                pass

            time.sleep(1. / 120)  # Env FPS, like pygame data_manager collection

        self.show_time_delays()
        return 0.

    @staticmethod
    def format_array(arr):
        return np.array2string(arr, formatter={'float_kind': lambda x: f"{x:.6f}"})

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
            "primary_rgb": camera_data['color1'],
            "gripper_rgb": camera_data['color2'],
            "primary_depth": camera_data['depth1'],
            "gripper_depth": camera_data['depth2'],
            "robot_obs": frame_robot_obs,
        }
        return env_obs_dict

    def step(self, action_D: np.ndarray) -> None:
        """ Send action to Robotic Arm """
        action = action_D.tolist()
        for i in range(3):
            if abs(action[i]) > 0.1:
                print("[Warning] bad action!!!", action[i])
                action[i] = max(min(float(action[i]), 1), -1.)
        action[0] *= self.speed_scale
        action[1] *= self.speed_scale
        action[2] *= self.speed_scale
        action[3] = min(float(action[3]), 1.)
        action[4] = min(float(action[4]), 1.)
        action[5] = min(float(action[5]), 1.)
        action[3] = max(float(action[3]), -1.)
        action[4] = max(float(action[4]), -1.)
        action[5] = max(float(action[5]), -1.)

        linear_xyz = {'x': action[0], 'y': action[1], 'z': action[2]}
        angular_xyz = {'x': action[3], 'y': action[4], 'z': action[5]}

        self.g = float(action[6])
        # if abs(action[3]) + abs(action[4]) + abs(action[5]) > abs(action[0]) + abs(action[1]) + abs(action[2]):
        #     self.robot.ang_jog_pub(angular_xyz)
        #     print("angular")
        # else:
        #     self.robot.linear_jog_pub(linear_xyz)
        #     print("linear")
        self.robot.lin_ang_jog_pub(
            linear_xyz,
            angular_xyz,
        )

    def on_gripper_move(self, g: float = None) -> None:
        self.g = g if g is not None else self.g
        self.robot.gripper_set_pub(self.g)

    def reset(self):
        # return
        # Sanity check for gripper
        # self.robot.gripper_set_pub(1)
        # time.sleep(0.5)
        self.robot.gripper_set_pub(0)
        time.sleep(1.5)

        # Go back home
        self.robot.joint_back_home()

    def stop(self):
        ## Just for coffee task
        # self.robot.gripper_set_pub(0)
        # time.sleep(1.5)
        # self.robot.joint_back_coffee()

        self.robot.stop()
        self.camera.stop()
        self.image_saver.save_to_video(self.image_save_fn)
        if self.action_save_flag:
            self.action_saver.save_to_video(self.action_save_fn)
