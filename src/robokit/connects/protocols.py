import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import copy

import numpy as np
import pydantic
from PIL import Image


class StepRequestFromEvaluator(pydantic.BaseModel):
    """
    Sent from evaluator to policy. All data are unnormalized/raw.

    Attributes:
        gt_video: (B,V*Ts,H,W,3) uint8
        tcp_state: (B,Ts,D) float32, optional
        stage_flag: 0:cold start, 1:hot start, can be used for online finetuning
        instruction: str
    """
    instruction: str
    stage_flag: int  # 0:cold start, 1:hot start

    gt_video: List[List[str]]  # len1=B, len2=V*Ts (V:#camera_views, Ts>=v1), each str is base64 of (H,W,rgb)
    num_camera_views: int = 1  # V, default=1
    tcp_state: Optional[List[List[List[float]]]] = None # len1=B, len2=Ts, len3=D, float32
    max_cache_action: int = None  # notified by the evaluator

    def decode_to_raw(self) -> Dict[str, Any]:
        def base64_to_image_H_W_C(img_base64: str) -> np.ndarray:
            img_byte = base64.b64decode(img_base64)
            img_pil = Image.open(io.BytesIO(img_byte), formats=["JPEG"])
            img_np = np.array(img_pil)  # (H,W,3) uint8
            return img_np

        def base64_to_video_T_H_W_C(vid_base64: List[str]) -> np.ndarray:
            return np.stack([base64_to_image_H_W_C(img_b64) for img_b64 in vid_base64], axis=0)

        gt_video_B_T_H_W_C = np.stack([base64_to_video_T_H_W_C(vid_b64) for vid_b64 in self.gt_video], axis=0)  # (B,Ts,H,W,3) uint8
        tcp_state_B_T_D = None if self.tcp_state is None else np.array(self.tcp_state, dtype=np.float32)  # (B,Ts,2) float32
        return {
            "instruction": self.instruction,
            "stage_flag": self.stage_flag,
            "gt_video": gt_video_B_T_H_W_C,
            "num_camera_views": self.num_camera_views,
            "tcp_state": tcp_state_B_T_D,
            "max_cache_action": self.max_cache_action
        }

    def decode_to_raw_buffer(self, out_video_buffer: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Decode request to raw numpy arrays.
        :param out_video_buffer: Optional pre-allocated buffer of shape (B, Ts, H, W, 3) and dtype uint8.
                                 If provided, data will be written into this buffer in-place.
        """
        B = len(self.gt_video)
        assert B != 0, "len(gt_video) should > 0"
        T = len(self.gt_video[0])
        assert T != 0, "len(gt_video[0]) should > 0"
        print(f"[DEBUG] decode_to_raw_buffer: B={B}, T={T}")

        if out_video_buffer is not None:
            # 使用外部传入的预分配内存
            gt_video_B_T_H_W_C = out_video_buffer
        else:
            # 兼容未传入 buffer 的情况：嗅探尺寸并临时分配
            first_img_byte = base64.b64decode(self.gt_video[0][0])
            with Image.open(io.BytesIO(first_img_byte), formats=["JPEG"]) as img_pil:
                first_img_np = np.array(img_pil)

            H, W, C = first_img_np.shape
            gt_video_B_T_H_W_C = np.empty((B, T, H, W, C), dtype=np.uint8)
            gt_video_B_T_H_W_C[0, 0] = first_img_np

        for b in range(B):
            for t in range(T):
                # 如果用户没传 buffer，且是第一帧，说明刚才嗅探时已经写过了，跳过
                if b == 0 and t == 0 and out_video_buffer is None:
                    continue
                img_byte = base64.b64decode(self.gt_video[b][t])
                with Image.open(io.BytesIO(img_byte), formats=["JPEG"]) as img_pil:
                    # np.array(img_pil) 会在局部产生一个极小的临时数组
                    # 随后立即被覆写进巨型预分配内存 gt_video_B_T_H_W_C 中
                    # 这将大对象的内存分配次数直接降为 0
                    gt_video_B_T_H_W_C[b, t] = np.array(img_pil)

        tcp_state_B_T_D = None if self.tcp_state is None else np.array(self.tcp_state, dtype=np.float32)

        return {
            "instruction": self.instruction,
            "stage_flag": self.stage_flag,
            "gt_video": gt_video_B_T_H_W_C,
            "num_camera_views": self.num_camera_views,
            "tcp_state": tcp_state_B_T_D,
            "max_cache_action": self.max_cache_action
        }

    @classmethod
    def encode_from_raw(cls,
                        instruction: str,
                        stage_flag: int,
                        gt_video: np.ndarray,  # (B,Ts,H,W,3) uint8
                        num_camera_views: int = 1,
                        tcp_state: Optional[np.ndarray] = None,  # (B,2) float32
                        max_cache_action: int = None
                        ) -> 'StepRequestFromEvaluator':
        """

        :param instruction: str
        :param stage_flag: int
        :param gt_video:  np.ndarray, (.,V*Ts,H,W,3) uint8
        :param num_camera_views: int
        :param tcp_state: np.ndarray, (.,Ts,D) float32
        :return:
        """
        instance = cls(
            instruction=instruction,
            stage_flag=stage_flag,
            gt_video=cls.video_np_to_base64(gt_video),
            num_camera_views=num_camera_views,
            tcp_state=None if tcp_state is None else tcp_state.tolist(),
            max_cache_action=max_cache_action,
        )
        return instance

    @staticmethod
    def create_zero_request():
        return StepRequestFromEvaluator.encode_from_raw(
            instruction="",
            stage_flag=0,
            gt_video=np.zeros((1, 10, 240, 320, 3), dtype=np.uint8),
            num_camera_views=1,
            tcp_state=np.zeros((1, 10, 8), dtype=np.float32),
            max_cache_action=0,
        )

    @staticmethod
    def video_np_to_base64(video: np.ndarray) -> List[List[str]]:
        assert video.ndim in [4, 5], f"video should be (B,Ts,H,W,3) or (Ts,H,W,3), got {video.shape}"
        assert video.dtype == np.uint8

        def image_to_base64(img_H_W_C) -> str:
            image_pil = Image.fromarray(img_H_W_C)
            image_bytes = io.BytesIO()
            image_pil.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return image_base64

        def video_to_base64(vid_T_H_W_C) -> List[str]:
            return [image_to_base64(img) for img in vid_T_H_W_C]

        if video.ndim == 4:  # No batch-dim
            video_base64 = [video_to_base64(video)]  # add batch-dim
        elif video.ndim == 5:
            video_base64 = [video_to_base64(vid) for vid in video]
        else:
            raise NotImplementedError

        return video_base64


class StepRequestFromPolicy(pydantic.BaseModel):
    """
    Sent from policy to evaluator. Action should be unnormalized/raw.

    Attributes:
        action: (B,v2,D) float32
        max_cache_action: int, optional, tell evaluator how many actions to cache, None means
    """

    action: List[List[List[float]]]  # len1=B, len2=v2 (v2=H), len3=D
    max_cache_action: int = None  # tell evaluator how many actions to cache, None means no limit

    def decode_to_raw(self) -> Dict[str, Any]:
        return {
            "action": np.array(self.action, dtype=np.float32),  # (B,v2,D) float32
            "max_cache_action": self.max_cache_action,
        }

    @classmethod
    def encode_from_raw(cls, action: np.ndarray) -> 'StepRequestFromPolicy':
        if action.ndim == 2:  # (H,D)
            action = action[np.newaxis, ...]  # add batch dim
        assert action.ndim == 3, f"action should be (B,H,D), got {action.shape}"
        assert action.dtype in [np.float32, np.float64], f"action should be float32 or float64, got {action.dtype}"
        instance = cls(action=action.tolist())
        return instance