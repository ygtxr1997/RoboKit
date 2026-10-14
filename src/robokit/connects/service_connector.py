import base64
import io
from typing import Optional, Sequence, List
import requests

import numpy as np
from PIL import Image

from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.data_manager.utils_multiview import cat_multiview_video_with_another
from robokit.debug_utils.times import time_stat


class FrameBuffer:
    def __init__(self, keys: List[str], max_obs_length: int):
        self.keys = keys
        self.max_obs_length = max_obs_length
        self.buffer = {key: [] for key in keys}

    def set_max_obs_length(self, max_obs_length: int):
        self.clear()
        self.max_obs_length = max_obs_length

    def add_single(self, **kwargs):
        used_keys = [k for k in self.keys if k in kwargs]
        unused_keys = [k for k in self.keys if k not in kwargs]
        unexpected_keys = [k for k in kwargs if k not in self.keys]
        if len(used_keys) == 0 or len(unused_keys) > 0:
            raise ValueError(f"[FrameBuffer] All keys must be provided when adding a single frame. "
                             f"Used keys: {used_keys}, Unused keys: {unused_keys}")
        if len(unexpected_keys) > 0:
            raise Warning(f"[FrameBuffer] Unexpected keys: {unexpected_keys}")

        for key in self.keys:
            self.buffer[key].append(kwargs[key])
            if len(self.buffer[key]) > self.max_obs_length:
                self.buffer[key].pop(0)

    def clear(self):
        for key in self.keys:
            self.buffer[key] = []

    def get_padded_frames(self, zero_pad: bool = False, concat_axis: int = 0) -> dict:
        """  Pad before the sequence to max_obs_length with zeros or first frame """
        padded_frames = {}
        for key in self.keys:
            frames = self.buffer[key]
            if len(frames) < self.max_obs_length:
                pad_length = self.max_obs_length - len(frames)
                pad_frame = np.zeros_like(frames[0]) if zero_pad else frames[0]
                frames = [pad_frame] * pad_length + frames
            padded_frames[key] = np.concatenate(frames, axis=concat_axis)
        return padded_frames


class ServiceConnector:
    def __init__(
            self,
            base_url: str = "http://localhost:6060",
            resize_hw: tuple = None,
            max_cache_actions: int = 28,
            num_camera_views: int = 1,
            close_online: bool = True,
    ):
        self.base_url = base_url
        self.http_session = requests.Session()
        self.resize_hw = resize_hw
        self.num_camera_views = num_camera_views
        self.close_online = close_online

        # Dynamic
        self.max_cache_actions = max_cache_actions
        self.task_instruction = None
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.stage_flag = 0  # 0:cold start, 1:hot start
        self.just_send = False  # just send frames or not ?

        print(f"[ServiceConnector] session created: {base_url}")

    def init_socket(self, task_instruction: str) -> int:
        resp = self.http_session.get(f"{self.base_url}/init")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        if isinstance(max_cache_action, list):
            max_cache_action = max_cache_action[0]
        self.max_cache_actions = min(max_cache_action, self.max_cache_actions)
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.task_instruction = task_instruction
        return max_cache_action

    def send_reset(self, task_instruction: str = None) -> int:
        resp = self.http_session.get(f"{self.base_url}/reset")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        if isinstance(max_cache_action, list):
            max_cache_action = max_cache_action[0]
        self.max_cache_actions = min(max_cache_action, self.max_cache_actions)
        self.send_cnt = 0

        self.cache_actions_B_T_D = None
        self.task_instruction = task_instruction
        return max_cache_action

    def send_obs_and_get_action(
            self,
            primary_rgb: np.ndarray,  # (B,T,H,W,C) uint8
            gripper_rgb: np.ndarray,  # (B,T,H,W,C) uint8
            task_description: Optional[str] = None,
            joint_state: np.ndarray = None,  # (B,T,D)
            *args, **kwargs
    ) -> np.ndarray:
        """

        :param primary_rgb: (B,T,H,W,C) uint8
        :param gripper_rgb: (B,T,H,W,C) uint8
        :param task_description:
        :param joint_state:  (B,T,D) float32
        :param args:
        :param kwargs:
        :return: actions predicted by the agent, (B,1,D)
        """
        if self.send_cnt % self.max_cache_actions == 0:
            from robokit.debug_utils.times import time_stat

            with time_stat("[DEBUG] ServiceConnector preparing data for sending"):
                joint_state = joint_state[:, :, ...] if joint_state is not None else None  # (B,T,D)
                gt_video = cat_multiview_video_with_another(
                    primary_rgb[:, :, ...],  # (B,T,H,W,C)
                    gripper_rgb[:, :, ...],  # (B,T,H,W,C)
                    sample_n_views=1,
                )
            with time_stat("[DEBUG] ServiceConnector resizing video"):
                # gt_video = self.resize_video(gt_video)
                pass  # No need to resize again, we resized in each frame loop

            print(f"[ServiceConnector] sending frames, i={self.send_cnt}, per={self.max_cache_actions}, "
                  f"i%per={self.send_cnt % self.max_cache_actions}, gt_video_shape={gt_video.shape}, "
                  f"joint_state_shape={joint_state.shape}")

            if joint_state.shape[-1] == 6:  # only contains TCP xyz + abc
                B, T, D = joint_state.shape
                joint_state = np.concatenate((joint_state, np.zeros((B, T, 2))), axis=-1)
            elif joint_state.shape[-1] == 12:  # TCP xyz-abc 6 + force 6
                pass
            assert joint_state.shape[-1] in (8, 12), "Joint state should be 8/12-dim."

            with time_stat("[DEBUG] ServiceConnector encode_from_raw"):
                request_to_policy = StepRequestFromEvaluator.encode_from_raw(
                    instruction=task_description,
                    stage_flag=self.stage_flag,  # NOTE: ori:0
                    gt_video=gt_video,
                    tcp_state=joint_state,
                    num_camera_views=self.num_camera_views,
                    max_cache_action=self.max_cache_actions,
                )

            sending_dict = request_to_policy.model_dump(mode="json")
            with time_stat("Post dict to GPU Inference"):
                response = self.http_session.post(
                    f"{self.base_url}/step",
                    json=sending_dict
                )
                response.raise_for_status()

            response = response.json()
            raw_actions = StepRequestFromPolicy(action=response["action"]).decode_to_raw()["action"]  # (B,H,7）

            self.cache_actions_B_T_D = raw_actions[:, :self.max_cache_actions, :]
            self.just_send = True
        else:  # don't send anything, to save robots bandwidth
            self.just_send = False

        ret_actions = self.cache_actions_B_T_D[:, self.send_cnt % self.max_cache_actions]
        ret_actions = np.array(ret_actions)[:, None]  # [B,D] -> (B,1,D)

        self.send_cnt += 1
        if not self.close_online:
            self.stage_flag = 1  # hot start
        return ret_actions

    @staticmethod
    # Resize images to target size
    def resize_image(img_array: np.ndarray, target_hw: tuple, is_depth: bool = False) -> np.ndarray:
        """
        Resize image array to target height and width
        :param img_array: (H, W, C) for RGB or (H, W) for depth
        :param target_hw: (target_height, target_width)
        :param is_depth: whether the image is depth map
        :return: resized image array
        """
        if img_array is None:
            return img_array
        pil_img = Image.fromarray(img_array)
        # Use NEAREST for depth to preserve values, BILINEAR for RGB
        resample_method = Image.NEAREST if is_depth else Image.BILINEAR
        resized_img = pil_img.resize(
            (target_hw[1], target_hw[0]),  # PIL uses (width, height)
            resample=resample_method
        )
        return np.array(resized_img)

    def resize_video(self, vid_array_B_T_H_W_C: np.ndarray, is_depth: bool = False) -> np.ndarray:
        """
        Resize video array to target height and width
        :param vid_array_B_T_H_W_C: (B, T, H, W, C) for RGB or (B, T, H, W) for depth
        :param is_depth: whether the video is depth map
        :return: resized video array
        """
        if vid_array_B_T_H_W_C is None or self.resize_hw is None:
            return vid_array_B_T_H_W_C

        original_shape = vid_array_B_T_H_W_C.shape
        B, T = original_shape[:2]
        target_h, target_w = self.resize_hw

        # 1. 提前计算好输入和输出的 Flat 形状，并预分配最终的连续内存块
        if vid_array_B_T_H_W_C.ndim == 5:  # RGB
            C = original_shape[-1]
            out_shape = (B, T, target_h, target_w, C)
            flat_out_shape = (B * T, target_h, target_w, C)
        elif vid_array_B_T_H_W_C.ndim == 4:  # Depth
            out_shape = (B, T, target_h, target_w)
            flat_out_shape = (B * T, target_h, target_w)
        else:
            raise ValueError(f"Expected 4D or 5D array, got {vid_array_B_T_H_W_C.ndim}D array")

        # 使用 np.empty 申请未初始化的内存，速度远快于 np.zeros，因为省去了填 0 的开销
        resized_array = np.empty(out_shape, dtype=vid_array_B_T_H_W_C.dtype)

        # 2. 获取输入和输出的扁平化视图 (View) —— 这里不产生任何数据拷贝
        flat_input = vid_array_B_T_H_W_C.reshape(-1, *original_shape[2:])
        flat_output = resized_array.reshape(flat_out_shape)

        with time_stat("[DEBUG] resize_video PIL loop"):
            # 3. 原地写回，彻底避免中间的 list 聚合和 np.stack 拷贝
            for i in range(B * T):
                flat_output[i] = ServiceConnector.resize_image(flat_input[i], self.resize_hw, is_depth)

        return resized_array


