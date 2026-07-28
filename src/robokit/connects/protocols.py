import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pydantic
from PIL import Image
import cv2


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

        from robokit.debug_utils.time_profiler import global_time_profiler as gtp
        # with gtp("pil_decode", group="process_request"):
        #     for b in range(B):
        #         for t in range(T):
        #             # 如果用户没传 buffer，且是第一帧，说明刚才嗅探时已经写过了，跳过
        #             if b == 0 and t == 0 and out_video_buffer is None:
        #                 continue
        #             img_byte = base64.b64decode(self.gt_video[b][t])
        #             with Image.open(io.BytesIO(img_byte), formats=["JPEG"]) as img_pil:
        #                 # np.array(img_pil) 会在局部产生一个极小的临时数组
        #                 # 随后立即被覆写进巨型预分配内存 gt_video_B_T_H_W_C 中
        #                 # 这将大对象的内存分配次数直接降为 0
        #                 gt_video_B_T_H_W_C[b, t] = np.array(img_pil)

        # with gtp("cv2_decode", group="process_request"):
        #     for b in range(B):
        #         for t in range(T):
        #             # 如果用户没传 buffer，且是第一帧，说明刚才嗅探时已经写过了，跳过
        #             if b == 0 and t == 0 and out_video_buffer is None:
        #                 continue
        #             img_byte = base64.b64decode(self.gt_video[b][t])
        #             img_1d = np.frombuffer(img_byte, dtype=np.uint8)
        #             img_bgr = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
        #             gt_video_B_T_H_W_C[b, t] = img_bgr[:, :, ::-1]

        def decode_one(job):
            b, t, s = job
            arr = np.frombuffer(base64.b64decode(s), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            return b, t, img

        jobs = [
            (b, t, self.gt_video[b][t])
            for b in range(B) for t in range(T)
            if not (b == 0 and t == 0 and out_video_buffer is None)
        ]

        with ThreadPoolExecutor(max_workers=min(T, (os.cpu_count() or 8) * 2)) as ex:
            for b, t, img_bgr in ex.map(decode_one, jobs, chunksize=16):
                gt_video_B_T_H_W_C[b, t] = img_bgr[:, :, ::-1]  # 或直接存BGR避免这步

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
        extra_vector: optional extra vector parameters, keyed by vector name. Each
            value is (B,v2,D) float32; B and v2 should match action, D may be different.
        max_cache_action: int, optional, tell evaluator how many actions to cache, None means
    """

    action: List[List[List[float]]]  # len1=B, len2=v2 (v2=H), len3=D
    extra_vector: Optional[Dict[str, List[List[List[float]]]]] = None  # name -> (B,v2,D) float32
    max_cache_action: int = None  # tell evaluator how many actions to cache, None means no limit

    @pydantic.model_validator(mode="after")
    def validate_extra_vector_shape(self):
        if self.extra_vector is None:
            return self

        action_B = len(self.action)
        action_v2 = len(self.action[0]) if action_B > 0 else 0
        for name, vector in self.extra_vector.items():
            if len(vector) != action_B:
                raise ValueError(f"extra_vector[{name}] B should match action, got {len(vector)} vs {action_B}")
            expected_D = None
            for b, vector_b in enumerate(vector):
                if len(vector_b) != action_v2:
                    raise ValueError(
                        f"extra_vector[{name}][{b}] v2 should match action, got {len(vector_b)} vs {action_v2}"
                    )
                for h, vector_bh in enumerate(vector_b):
                    current_D = len(vector_bh)
                    if expected_D is None:
                        expected_D = current_D
                    elif current_D != expected_D:
                        raise ValueError(
                            f"extra_vector[{name}] should have consistent D, got {current_D} at [{b}][{h}] "
                            f"vs {expected_D}"
                        )
        return self

    def decode_to_raw(self) -> Dict[str, Any]:
        return {
            "action": np.array(self.action, dtype=np.float32),  # (B,v2,D) float32
            "extra_vector": None if self.extra_vector is None else {
                name: np.array(vector, dtype=np.float32)
                for name, vector in self.extra_vector.items()
            },
            "max_cache_action": self.max_cache_action,
        }

    @classmethod
    def encode_from_raw(cls,
                        action: np.ndarray,
                        extra_vector: Optional[Dict[str, Any]] = None,
                        max_cache_action: Optional[int] = None,
                        ) -> 'StepRequestFromPolicy':
        if action.ndim == 2:  # (H,D)
            action = action[np.newaxis, ...]  # add batch dim
        assert action.ndim == 3, f"action should be (B,H,D), got {action.shape}"
        assert action.dtype in [np.float32, np.float64], f"action should be float32 or float64, got {action.dtype}"

        extra_vector_json = None
        if extra_vector is not None:
            assert isinstance(extra_vector, dict), "extra_vector should be a dict: name -> (B,H,D)"
            extra_vector_json = {}
            for name, vector in extra_vector.items():
                assert isinstance(name, str), f"extra_vector name should be str, got {type(name)}"
                vector = np.asarray(vector)
                if vector.ndim == 2:  # (H,D)
                    vector = vector[np.newaxis, ...]  # add batch dim
                assert vector.ndim == 3, f"extra_vector[{name}] should be (B,H,D), got {vector.shape}"
                assert vector.shape[:2] == action.shape[:2], \
                    f"extra_vector[{name}] B,H should match action, got {vector.shape[:2]} vs {action.shape[:2]}"
                assert vector.dtype in [np.float32, np.float64], \
                    f"extra_vector[{name}] should be float32 or float64, got {vector.dtype}"
                extra_vector_json[name] = vector.tolist()

        instance = cls(
            action=action.tolist(),
            extra_vector=extra_vector_json,
            max_cache_action=max_cache_action,
        )
        return instance
