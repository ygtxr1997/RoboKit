import time
import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np
from PIL import Image

from robokit.connects.service_connector import ServiceConnector
from robokit.robots.robot_client_inovo import RobotClient
from robokit.data_manager.realsense_handler import RealsenseHandler
from robokit.data_manager.ft300_handler import FT300Handler
from robokit.data_manager.data_handler import DataHandler, ImageAsVideoSaver, ActionAsVideoSaver
from robokit.debug_utils.images import concatenate_rgb_images
from robokit.debug_utils.times import time_stat


class ImageRingBuffer:
    def __init__(self, capacity=24, shape=(480, 640, 3), dtype=np.uint8):
        self.capacity = capacity
        # 初始化时一次性分配好所有内存
        self.buffer = np.zeros((capacity, *shape), dtype=dtype)
        self.idx = 0
        self.is_full = False

    def append(self, frame):
        # 原地覆写最旧的数据，不产生新的内存分配
        self.buffer[self.idx] = frame
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.is_full = True

    def get_ordered_stack(self):
        # 如果需要严格的时间顺序（最旧到最新）
        if not self.is_full:
            return self.buffer[:self.idx]
        # np.roll 返回的是视图的重新排列，如果必须连续内存才拷贝
        # 但很多操作直接对 roll 后的结果操作即可
        return np.roll(self.buffer, -self.idx, axis=0)


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
                 future_skip: int = 1,  # 1 means no skip, not supported yet, will be handled in GPU api side
                 enable_auto_ae_wb: bool = True,
                 buffer_size: int = 1,
                 speed_scale: float = 0.75,
                 go_home_at_begin: bool = True,
                 reset_remote_api_at_begin: bool = True,
                 sleep_frames_after_send: float = 0.,
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
        self.future_skip = future_skip
        self.go_home_at_begin = go_home_at_begin
        self.reset_remote_api_at_begin = reset_remote_api_at_begin
        self.start_time = time.time()

        # Cameras
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

        self.ftsensor = None  # create and calibrate after go_home in run()

        self.speed_scale = speed_scale
        self.sleep_frames_after_send = sleep_frames_after_send

        # Dynamic variables
        self.step_cnt = 0
        self.g = 0.
        self.frame_buffer: Dict[str, ImageRingBuffer] = {
            'primary_rgb': ImageRingBuffer(capacity=buffer_size, shape=(resize_hw[0], resize_hw[1], 3)),
            'gripper_rgb': ImageRingBuffer(capacity=buffer_size, shape=(resize_hw[0], resize_hw[1], 3)),
            'joint_state': ImageRingBuffer(capacity=buffer_size, shape=(12,), dtype=np.float32),
        }  # some policies require 2 or more observation frames
        self.buffer_size = buffer_size

        # Record
        self.log_time_delays = []
        self.image_save_speedup = image_save_speedup
        self.image_save_fn = image_save_fn
        self.image_saver = ImageAsVideoSaver(
            buffer_size=run_loops,
            frame_rate=int(self.fps * self.image_save_speedup),
            height=int(img_hw[0] / 1.5),  # TODO: resize to save memory
            width=int(img_hw[1] / 1.5),  # no need to concat for multi-view image saver
            n_cameras=2,
        )
        self.action_save_flag = action_save_flag
        self.action_save_fn = image_save_fn.replace('.mp4', '_action.mp4')
        self.action_saver = ActionAsVideoSaver(
            buffer_size=run_loops,
            frame_rate=int(self.fps * self.image_save_speedup),
            height=4.5,
            width=6,
        )

        # FPS statistics
        self.first_frame_time = None  # 第一帧的时间戳
        self.last_frame_time = None  # 上一帧的时间戳
        self.total_frames = 0  # 总帧数

    def time_tick(self):
        self.start_time = time.time()

    def time_tok(self):
        current_time = time.time()
        time_delay = current_time - self.start_time
        print(f"[DebugEvaluator] time cost: {time_delay * 1000.:.1f} ms")
        self.log_time_delays.append(time_delay)

        # 计算实际帧率
        if self.first_frame_time is None:
            self.first_frame_time = current_time

        if self.last_frame_time is not None:
            # 计算瞬时帧率（基于上一帧）
            instant_interval = current_time - self.last_frame_time
            instant_fps = 1.0 / instant_interval if instant_interval > 0 else 0

            # 计算平均帧率（基于所有帧）
            total_elapsed = current_time - self.first_frame_time
            avg_fps = self.total_frames / total_elapsed if total_elapsed > 0 else 0

            print(
                f"[FPS] Instant: {instant_fps:.2f} FPS, Average: {avg_fps:.2f} FPS (target: {self.fps} FPS, frames: {self.total_frames})")

        self.last_frame_time = current_time
        self.total_frames += 1

    def show_time_delays(self):
        delays = np.array(self.log_time_delays) * 1000.  # second -> ms
        fps_lists = (1000. / (delays + 1e-6)).astype(np.int32)
        print(f"[DebugEvaluator] time delay: min={np.min(delays):.1f} ms, max={np.max(delays):.1f} ms, "
              f"avg={np.average(delays):.1f} ms; "
              f"min_fps={np.min(fps_lists)} FPS, max_fps={np.max(fps_lists)} FPS, "
              f"avg_fps={np.average(fps_lists)} FPS.")

    def run(self) -> float:
        target_interval = 1. / self.fps  # 目标时间间隔
        last_execution_time = 0  # 上次执行时间
        max_inference_time = 0.

        cur_task_text = self.cur_task_text
        buffer_size = self.buffer_size
        self.reset()
        if self.reset_remote_api_at_begin:
            self.connector.init_socket(cur_task_text)
            self.connector.send_reset(cur_task_text)

        # Force Sensor
        self.ftsensor = FT300Handler()
        if not self.ftsensor.connect(calibrate_samples=200):
            print("Failed to connect force torque sensor")
            exit()

        for _ in tqdm(range(100), desc="Skipping frames:"):
            frame_obs = self.capture_env_observation()

        while True:
            cycle_start_time = time.time()
            self.time_tick()

            # 1. Get current scene observation (fast)
            obs_start = time.time()
            frame_obs = self.capture_env_observation()
            obs_time = time.time() - obs_start
            # print(f"[Debug] Observation capture time: {obs_time * 1000:.1f}ms")

            # 2. 处理帧缓存（快速）
            def resize_frame(frame):
                if self.connector.resize_hw is not None:
                    return np.array(Image.fromarray(frame).resize((self.connector.resize_hw[1], self.connector.resize_hw[0])))
                return frame

            cur_primary_rgb = resize_frame(frame_obs['primary_rgb'])
            cur_gripper_rgb = resize_frame(frame_obs['gripper_rgb'])
            # cur_joint_state = frame_obs['robot_obs'].tolist()[7:13]  # Real joint_state
            cur_joint_state = frame_obs['robot_obs'][:6]  # TCP pos as joint_state
            cur_force = frame_obs['force']  # (6,)
            cur_joint_state = np.concatenate([cur_joint_state, cur_force], axis=0)  # (12,)

            # --- 更新帧缓存 ---
            with time_stat("[DEBUG] Frame Buffer Update"):
                for key, val in zip(
                        ['primary_rgb', 'gripper_rgb', 'joint_state'],
                        [cur_primary_rgb, cur_gripper_rgb, cur_joint_state]
                ):
                    self.frame_buffer[key].append(val)

            # Save current image
            # saved_image = concatenate_rgb_images(
            #     self.frame_buffer['primary_rgb'][-1],
            #     self.frame_buffer['gripper_rgb'][-1],
            # )  # (H, 2*W, C)
            with time_stat("[DEBUG] concat videos"):
                saved_image_VT_H_W_C = np.concatenate([
                    self.frame_buffer['primary_rgb'].get_ordered_stack()[-1:, :],  # (N,H,W,C) -> (1,H,W,C)
                    self.frame_buffer['gripper_rgb'].get_ordered_stack()[-1:, :],  # (N,H,W,C) -> (1,H,W,C)
                ], axis=0)  # (2,H,W,C)
                self.image_saver.add_video(saved_image_VT_H_W_C)  # last is current

            with time_stat("[DEBUG] send_obs_and_get_action"):
                # 3. GPU推理（最慢：6秒左右）
                inference_start = time.time()
                action = self.connector.send_obs_and_get_action(
                    primary_rgb=self.frame_buffer['primary_rgb'].get_ordered_stack()[None, :],  # (1,T,H,W,C) to List[str]
                    gripper_rgb=self.frame_buffer['gripper_rgb'].get_ordered_stack()[None, :],
                    task_description=cur_task_text,
                    joint_state=self.frame_buffer['joint_state'].get_ordered_stack()[None, :],  # (B,T,6+6)
                )
                inference_time = time.time() - inference_start
                max_inference_time = max(max_inference_time, inference_time)
                print(f"[Debug] GPU inference time: {inference_time * 1000:.1f}ms, max={max_inference_time * 1000:.1f}ms")

                assert action.shape[-1] == 7, f"action.shape[-1]={action.shape} is not (:,7)"

            ## Slow down when the height is too low
            # safe_min_z = 0.2
            # print(cur_joint_state[-1][2], safe_min_z)
            # if cur_joint_state[-1][2] <= safe_min_z:
            #     for h in range(action.shape[0]):
            #         if action[h, 2] < 0.:
            #             action[h, 2] = 0.7 * action[h, 2]

            ## Disable rotation at `x` and `y` axis
            # for h in range(action[0].shape[0]):
            #     action[0, h, 3] = 0.
            #     action[0, h, 4] = 0.

            # Fallback when force fz is too large
            fz = frame_obs['force'][2]
            if fz < -14.5:  # 过大说明可能撞到了, 给 z 加一个反向的动作补偿
                print("[Debug] force fz:", fz)
                for h in range(action[0].shape[0]):
                    if action[0, h, 2] < 0.:
                        action[0, h, 2] = -0.1 * action[0, h, 2]

            # 4. Conduct action in real-world environment, FPS control
            current_time = time.time()
            cycle_elapsed = current_time - cycle_start_time  # 本轮已用时间
            if last_execution_time > 0:  # 非首次执行
                time_since_last_execution = current_time - last_execution_time

                if time_since_last_execution < target_interval:
                    sleep_time = target_interval - time_since_last_execution
                    print(f"[Info] Sleeping {sleep_time * 1000:.1f}ms to maintain {self.fps} FPS")
                    time.sleep(sleep_time)
                elif time_since_last_execution > target_interval * 2:
                    print(
                        f"[Warning] Execution interval too long: "
                        f"{time_since_last_execution:.2f}s (target: {target_interval * 1000:.1f}ms)")

            # 5. 执行动作（相对快速）
            execution_start = time.time()
            last_execution_time = execution_start

            print("[Info]", self.step_cnt, cur_task_text[:20], f"sleep_frames: {self.sleep_frames_after_send}")

            with time_stat("Robot Action Execution"):
                self.step(action_D=action[0, 0, :])
                self.on_gripper_move()
                if self.connector.just_send:
                    time.sleep(self.sleep_frames_after_send * 1. / self.fps)  # TODO: force to wait the action has been executed
                    self.step(action_D=np.zeros_like(action[0, 0, :]))  # 停止动作，保持在当前状态
            self.action_saver.add_action(action[0, 0, :])

            execution_time = time.time() - execution_start
            total_cycle_time = time.time() - cycle_start_time

            # Robot arm conducting delay
            self.time_tok()
            self.step_cnt += 1
            time_stat.print_stats()

            if self.step_cnt >= self.run_loops:
                break  # Finished

        self.show_time_delays()
        return 0.

    @staticmethod
    def format_array(arr):
        return np.array2string(arr, formatter={'float_kind': lambda x: f"{x:.6f}"})

    def capture_env_observation(self) -> dict:
        # camera_data = self.camera.capture_frames()
        robot_data_time = time.time()
        robot_data = self.robot.get_current_frame_info()  # very fast

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

        with time_stat("Force Sensor"):
            ft_data = self.ftsensor.read_ft()  # (fx, fy, fz, mx, my, mz)
            frame_force = np.array([
                ft_data[0], ft_data[1], ft_data[2],
                ft_data[3], ft_data[4], ft_data[5],
            ])  # (6,)

        with time_stat("Multi-view Cameras"):
            camera_data = self.camera.capture_frames()

        env_obs_dict = {
            "primary_rgb": camera_data['color1'],
            "gripper_rgb": camera_data['color2'],
            "primary_depth": camera_data['depth1'],
            "gripper_depth": camera_data['depth2'],
            "robot_obs": frame_robot_obs,
            "force": frame_force,
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
        ## TODO: why this?
        # action[3] = min(float(action[3]), 1.)
        # action[4] = min(float(action[4]), 1.)
        # action[5] = min(float(action[5]), 1.)
        # action[3] = max(float(action[3]), -1.)
        # action[4] = max(float(action[4]), -1.)
        # action[5] = max(float(action[5]), -1.)

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
        if self.go_home_at_begin:
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
