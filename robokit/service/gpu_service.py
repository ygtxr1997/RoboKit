import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict

import pydantic
from PIL import Image
from fastapi import FastAPI

from robokit.debug_utils.debug_classes import DebugModel


""" How to use me?
$ CUDA_VISIBLE_DEVICES=9 uvicorn gpu_service:gpu_app --port 6060
"""
gpu_app = FastAPI()


class StepRequestWithObservation(pydantic.BaseModel):
    primary_rgb: str
    gripper_rgb: str
    instruction: str
    joint_states: List[float]


@lru_cache()
def get_agent(device: str):
    model = DebugModel(sleep_duration=100)
    return model


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/reset")
def model_reset():
    agent = get_agent("cuda")
    agent.reset()
    return {"message": "Done"}


@gpu_app.post("/step")
def model_step(step_request: StepRequestWithObservation):
    agent = get_agent("cuda")
    print("[gpu_service] Using cached ckpt from: None. Model type:", type(agent))

    # 1. Decode observation from received request
    primary_img = base64.b64decode(step_request.primary_rgb)
    primary_img = Image.open(io.BytesIO(primary_img), formats=["JPEG"])

    gripper_img = base64.b64decode(step_request.gripper_rgb)
    gripper_img = Image.open(io.BytesIO(gripper_img), formats=["JPEG"])

    instruction_text = step_request.instruction
    joint_states = step_request.joint_states

    # 2. Preprocess, e.g resize, normalize, to_tensor, to_device
    pass
    obs = {
        "rgb_obs": {
            "rgb_static": primary_img,
            "rgb_gripper": gripper_img,
        },
    }
    cond = {
        "lang_text": instruction_text,
        "proprioception": joint_states,
    }

    # 3. Model inference
    action = agent.step(obs, cond)

    print("[gpu_service] Action:", len(action))

    # 4. Return results
    return {"action": action}
