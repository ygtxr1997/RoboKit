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
    ):
        self.base_url = base_url
        self.http_session = requests.Session()

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

    def send_reset(self, task_instruction: str) -> int:
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
            joint_state: Optional[Sequence] = None,
            *args, **kwargs
    ) -> np.ndarray:
        """

        :param primary_rgb:
        :param gripper_rgb:
        :param task_description:
        :param joint_state:
        :param args:
        :param kwargs:
        :return: actions predicted by the agent, (B,1,D)
        """
        if self.send_cnt % self.max_cache_actions == 0:
            print(f"[ServiceConnector] sending frames, i={self.send_cnt}, per={self.max_cache_actions}, "
                  f"i%per={self.send_cnt % self.max_cache_actions}")
            request_to_policy = StepRequestFromEvaluator.encode_from_raw(
                instruction=task_description,
                stage_flag=0,
                gt_video=cat_multiview_video_with_another(
                    primary_rgb,
                    gripper_rgb,
                    sample_n_views=1,
                ),
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
    def img_np_to_base64(image: np.ndarray) -> List[str]:
        assert image.dtype == np.uint8

        def image_to_base64(img):
            image_pil = Image.fromarray(img)
            image_bytes = io.BytesIO()
            image_pil.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return image_base64

        if image.ndim == 3:  # Single frame
            image_base64 = [image_to_base64(image)]
        else:
            assert image.ndim == 4  # Multi frames
            image_base64 = [image_to_base64(img) for img in image]

        return image_base64


