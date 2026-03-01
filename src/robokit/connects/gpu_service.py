import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from abc import abstractmethod, ABC

import numpy as np
from fastapi import FastAPI

import torch
from torchvision.transforms import transforms

from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.data_manager.utils_multiview import get_frames_from_multiview_video, cat_multiview_video_with_zeros
from robokit.debug_utils.debug_classes import ReplayModel


""" How to use me?
On `dongxu-g6.cs.hku.hk`, assuming your port is `6X60`, then run:
>$ CUDA_VISIBLE_DEVICES={GPU_INDEX} uvicorn gpu_service:gpu_app --port '6X60'
"""
class GPUServiceAPI(ABC):
    def __init__(self):
        self.model = None
        self.config = None

        self.max_cache_action = 10  # will notify evaluator the max length

    @abstractmethod
    def get_model_and_others(self, device: str) -> Dict[str, Any]:
        """
        Example:
        config = {
            'image_shape': (3,256,256),
            'data_min': [-1]*7,
            'data_max': [1.]*7,
            'infer_per_steps": 4}
        ## Op1. Debug model, sleep only
        # model = DebugModel(sleep_duration=100)
        ## Op2. Replay model, load action data_manager and sleep
        model = ReplayModel(sleep_duration=25,
                            replay_root="/home/geyuan/datasets/TCL/collected_data")
        model = model.to(device).eval()
        self.model = model
        self.config = config
        return {"model": model, "config": config}
        """
        pass

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def reset_model(self):
        """
        Example:
        self.model.reset()
        """
        pass

    @abstractmethod
    def predict_action(self, obs_dict: Dict[str, Any]) -> Union[np.ndarray, Dict[str, Any]]:
        pass

    @abstractmethod
    def denorm_action(self, action: np.ndarray) -> np.ndarray:
        pass


gpu_app = FastAPI()
gpu_service_api = GPUServiceAPI()

@lru_cache()
def get_agent(device: str):
    model_and_others = gpu_service_api.get_model_and_others(device)
    return model_and_others


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    gpu_service_api.init_model()
    return {"message": "Initialized.", "max_cache_action": gpu_service_api.max_cache_action}

@gpu_app.get("/reset")
def model_reset():
    gpu_service_api.reset_model()
    return {"message": "Done", "max_cache_action": gpu_service_api.max_cache_action}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    agent_and_others = get_agent("cuda")
    agent = agent_and_others['model']
    config = agent_and_others['config']
    print("[gpu_service] Using cached ckpt from: None. Model type:", type(agent))

    # 1. Decode observation from received request
    step_data = step_request.decode_to_raw()
    instruction_text = step_data["instruction"]
    stage_flag = step_data["stage_flag"]
    gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8, Ts can be larger than v1
    tcp_state = step_data["tcp_state"]  # (B,Ts,D) float32 or None

    # 2. Preprocess, e.g resize, normalize, to_tensor, to_device
    primary_img = torch.from_numpy(gt_video)  # (B,V*T,C,H,W)
    primary_img = primary_img.to("cuda")  # (B,T,C,H,W)
    obs_dict = {
        "image": primary_img,  # should be (B,T,C,H,W)
    }

    # 3. Model inference
    with torch.no_grad():
        action = gpu_service_api.predict_action(obs_dict)['action_pred']
    print("[gpu_service] action:", action.shape, action.min(dim=0)[0], action.max(dim=0)[0])
    action = action.cpu().numpy()  # (B,T,D)

    # 4. Send action request
    request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=action)
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    # DEBUG
    import numpy as np
    from robokit.connects.service_connector import ServiceConnector

    agent = get_agent("cuda")

    zero_rgb = np.zeros((2, 480, 848, 3), dtype=np.uint8)  # (T,H,W,C)

    debug_request = StepRequestFromEvaluator(
        primary_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        gripper_rgb="none",
        instruction="none",
        joint_states=[0.] * 6,
    )
    pred_action = model_step(debug_request)
    print(pred_action)
