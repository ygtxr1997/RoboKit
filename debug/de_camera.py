import pyrealsense2 as rs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from robokit.data.realsense_handler import RealsenseHandler


rs_handler = RealsenseHandler()
rs_handler.capture_frames(skip_seconds=3)

wbs = range(0, 10, 1)
for wb in wbs:
    rs_handler.capture_frames(save_prefix=f"tmpwb={wb}_close", skip_frames=100)

rs_handler.stop()

print("OK")
