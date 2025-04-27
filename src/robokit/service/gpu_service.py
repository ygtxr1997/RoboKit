import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict
from abc import abstractmethod, ABC

import pydantic
from PIL import Image
from fastapi import FastAPI

import torch
from torchvision.transforms import transforms

from robokit.debug_utils.debug_classes import ReplayModel


""" How to use me?
On `dongxu-g6.cs.hku.hk`, assuming your port is `6X60`, then run:
>$ CUDA_VISIBLE_DEVICES={GPU_INDEX} uvicorn gpu_service:gpu_app --port '6X60'
"""

class StepRequestWithObservation(pydantic.BaseModel):
    primary_rgb: List[str]
    gripper_rgb: List[str]
    instruction: str
    joint_states: List[float]


class GPUServiceAPI(ABC):
    def __init__(self):
        self.model = None
        self.config = None

    @abstractmethod
    def get_model_and_config(self, device: str):
        """
        Example:
        config = {
            'image_shape': (3,256,256),
            'data_min': [-1]*7,
            'data_max': [1.]*7,
            'infer_per_steps": 4}
        ## Op1. Debug model, sleep only
        # model = DebugModel(sleep_duration=100)
        ## Op2. Replay model, load action data and sleep
        model = ReplayModel(sleep_duration=25,
                            replay_root="/home/geyuan/datasets/TCL/collected_data")
        model = model.to(device).eval()
        self.model = model
        self.config = config
        return model, config
        """
        pass

    @abstractmethod
    def reset_model(self):
        """
        Example:
        self.model.reset()
        """
        pass

gpu_app = FastAPI()
gpu_service_api = GPUServiceAPI()


@lru_cache()
def get_agent(device: str):
    model, cfg = gpu_service_api.get_model_and_config(device)
    return model, cfg


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/reset")
def model_reset():
    gpu_service_api.reset_model()
    return {"message": "Done"}


@gpu_app.post("/step")
def model_step(step_request: StepRequestWithObservation):
    agent, config = get_agent("cuda")
    print("[gpu_service] Using cached ckpt from: None. Model type:", type(agent))

    # 1. Decode observation from received request
    image_shape = config.image_shape
    primary_imgs = []
    for primary_img in step_request.primary_rgb:
        primary_img = base64.b64decode(primary_img)
        primary_img = Image.open(io.BytesIO(primary_img), formats=["JPEG"])

        rgb_transform = transforms.Compose([
            transforms.Resize(image_shape[1:]),
            transforms.ToTensor(),
        ])
        primary_img = rgb_transform(primary_img) * 2. - 1.  # (C,H,W), in [-1,1]
        primary_imgs.append(primary_img)

    # gripper_img = base64.b64decode(step_request.gripper_rgb)
    # gripper_img = Image.open(io.BytesIO(gripper_img), formats=["JPEG"])

    instruction_text = step_request.instruction
    joint_states = step_request.joint_states

    # 2. Preprocess, e.g resize, normalize, to_tensor, to_device
    primary_img = torch.stack(primary_imgs, dim=0)  # (T,C,H,W)
    # print(primary_img.shape)
    primary_img = primary_img.to("cuda").unsqueeze(0)  # (B,T,C,H,W)
    obs_dict = {
        "image": primary_img,  # should be (B,T,C,H,W)
    }
    # cond = {
    #     "lang_text": instruction_text,
    #     "proprioception": joint_states,
    # }

    # 3.a Model inference
    infer_per_steps = config.infer_per_steps
    action_idx = agent.infer_frame_idx % infer_per_steps
    data_min, data_max = config.data_min, config.data_max
    if action_idx == 0:
        with torch.no_grad():
            action = agent.predict_action(obs_dict)['action_pred']
            action = action[0, :]  # remove batch_dim

        # 3.b Postprocess
        print(action.shape, action.min(dim=0)[0], action.max(dim=0)[0])
        action = (action * 0.5 + 0.5).cpu()  # in [0,1]
        action = action * (data_max - data_min) + data_min
        # print(action.shape, action.min(dim=0), action.max(dim=0))
        agent.cache_action = action
    else:
        action = agent.cache_action

    # 4. Return results
    frame_action = action[action_idx].numpy().tolist()
    if frame_action[6] > 0.5:
        frame_action[6] = 1.
    else:
        frame_action[6] = 0.
    print("[gpu_service] Action:", len(frame_action), frame_action)

    agent.infer_frame_idx += 1
    return {"action": frame_action}


if __name__ == "__main__":
    # DEBUG
    import numpy as np
    from robokit.service.service_connector import ServiceConnector

    agent = get_agent("cuda")

    zero_rgb = np.zeros((2, 480, 848, 3), dtype=np.uint8)  # (T,H,W,C)

    debug_request = StepRequestWithObservation(
        primary_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        gripper_rgb="none",
        instruction="none",
        joint_states=[0.] * 6,
    )
    pred_action = model_step(debug_request)
    print(pred_action)
