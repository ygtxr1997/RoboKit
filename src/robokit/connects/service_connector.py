import base64
import io
from typing import Optional, Sequence, List
import requests

import numpy as np
from PIL import Image

from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.data_manager.utils_multiview import cat_multiview_video_with_another


class ServiceConnector:
    def __init__(
            self,
            base_url: str = "http://localhost:6060",
            resize_hw: tuple = None,
    ):
        self.base_url = base_url
        self.http_session = requests.Session()
        self.resize_hw = resize_hw

        # Dynamic
        self.max_cache_actions = 1
        self.task_instruction = None
        self.send_cnt = 0
        self.cache_actions_B_T_D = None

        print(f"[ServiceConnector] session created: {base_url}")

    def init_socket(self, task_instruction: str) -> int:
        resp = self.http_session.get(f"{self.base_url}/init")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        self.max_cache_actions = max(max_cache_action, 1)
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.task_instruction = task_instruction
        return max_cache_action

    def send_reset(self, task_instruction: str = None) -> int:
        resp = self.http_session.get(f"{self.base_url}/reset")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        self.max_cache_actions = max(max_cache_action, 1)
        self.send_cnt = 0

        self.cache_actions_B_T_D = None
        self.task_instruction = task_instruction
        return max_cache_action

    def send_obs_and_get_action(
            self,
            primary_rgb: np.ndarray,  # (B,T,H,W,C) uint8
            gripper_rgb: np.ndarray,  # (B,T,H,W,C) uint8
            task_description: Optional[str] = None,
            joint_state: np.ndarray = None,
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
            print(f"[ServiceConnector] sending frames, i={self.send_cnt}, per={self.max_cache_actions}, "
                  f"i%per={self.send_cnt % self.max_cache_actions}")
            gt_video = cat_multiview_video_with_another(
                primary_rgb,
                gripper_rgb,
                sample_n_views=1,
            )
            gt_video = self.resize_video(gt_video)
            if joint_state.shape[-1] == 6:  # only contains TCP xyz + abc
                B, T, D = joint_state.shape
                joint_state = np.concatenate((joint_state, np.zeros((B, T, 2))), axis=-1)
            assert joint_state.shape[-1] == 8, "Joint state should be 8-dim."

            request_to_policy = StepRequestFromEvaluator.encode_from_raw(
                instruction=task_description,
                stage_flag=0,
                gt_video=gt_video,
                tcp_state=joint_state,
            )

            sending_dict = request_to_policy.model_dump(mode="json")
            response = self.http_session.post(
                f"{self.base_url}/step",
                json=sending_dict
            )
            response.raise_for_status()

            response = response.json()
            raw_actions = StepRequestFromPolicy(action=response["action"]).decode_to_raw()["action"]  # (B,H,7）

            self.cache_actions_B_T_D = raw_actions
        else:  # don't send anything, to save robots bandwidth
            pass

        ret_actions = self.cache_actions_B_T_D[:, self.send_cnt % self.max_cache_actions]
        ret_actions = np.array(ret_actions)[:, None]  # [B,D] -> (B,1,D)

        self.send_cnt += 1
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

        # Get original shape
        original_shape = vid_array_B_T_H_W_C.shape

        # Reshape to (B*T, H, W, C) or (B*T, H, W) for batch processing
        if vid_array_B_T_H_W_C.ndim == 5:  # RGB: (B, T, H, W, C)
            B, T = original_shape[:2]
            reshaped = vid_array_B_T_H_W_C.reshape(-1, *original_shape[2:])  # (B*T, H, W, C)
        elif vid_array_B_T_H_W_C.ndim == 4:  # Depth: (B, T, H, W)
            B, T = original_shape[:2]
            reshaped = vid_array_B_T_H_W_C.reshape(-1, *original_shape[2:])  # (B*T, H, W)
        else:
            raise ValueError(f"Expected 4D or 5D array, got {vid_array_B_T_H_W_C.ndim}D array")

        # Resize all frames
        target_hw = self.resize_hw
        resized_frames = [
            ServiceConnector.resize_image(frame, target_hw, is_depth)
            for frame in reshaped
        ]
        resized_array = np.stack(resized_frames, axis=0)

        # Reshape back to (B, T, target_h, target_w, C) or (B, T, target_h, target_w)
        if vid_array_B_T_H_W_C.ndim == 5:
            return resized_array.reshape(B, T, target_hw[0], target_hw[1], original_shape[-1])
        else:
            return resized_array.reshape(B, T, target_hw[0], target_hw[1])


