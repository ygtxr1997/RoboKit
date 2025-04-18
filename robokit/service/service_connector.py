import base64
import io
from typing import Optional, Sequence
import requests


import numpy as np
from PIL import Image


class ServiceConnector:
    def __init__(
            self,
            base_url: str = "http://localhost:6060",
    ):
        self.base_url = base_url
        self.http_session = requests.Session()

        print(f"[ServiceConnector] session created: {base_url}")

    def reset(self, task_description: str) -> None:
        pass

        resp = self.http_session.get(f"{self.base_url}/reset")
        resp.raise_for_status()

    def step(
            self,
            primary_rgb: np.ndarray,
            gripper_rgb: np.ndarray,
            task_description: Optional[str] = None,
            joint_states: Optional[Sequence] = None,
            *args, **kwargs
    ) -> np.ndarray:
        """
        Input: information sent by the environment
        Output: actions predicted by the agent
        """
        primary_base64 = self.img_np_to_base64(primary_rgb)
        gripper_base64 = self.img_np_to_base64(gripper_rgb)

        response = self.http_session.post(
            f"{self.base_url}/step",
            json={
                "primary_rgb": primary_base64,
                "gripper_rgb": gripper_base64,
                "instruction": task_description,
                "joint_states": joint_states,
            }
        )
        response.raise_for_status()

        response = response.json()
        raw_actions = np.array(response["action"])[None]  # (1,6)
        return raw_actions

    @staticmethod
    def img_np_to_base64(image: np.ndarray) -> str:
        assert image.dtype == np.uint8

        image_pil = Image.fromarray(image)
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return image_base64


