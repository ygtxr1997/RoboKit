import time
from typing import List
import numpy as np
from PIL import Image

from robokit.service.service_connector import ServiceConnector


class DebugModel:
    """ Run on GPU server. """
    def __init__(self, sleep_duration: int = 100):
        self.sleep_duration = sleep_duration  # in ms

    def __call__(self, *args, **kwargs) -> List[float]:
        start = time.time()
        time.sleep(self.sleep_duration / 1000.0)  # in seconds
        print("Sleeping for: ", int((time.time() - start) * 1000))
        action = np.zeros((1, 6), dtype=np.float32)
        return action[0, :].tolist()

    def step(self, *args, **kwargs) -> List[float]:
        return self(*args, **kwargs)

    def reset(self):
        pass


class DebugEvaluator:
    """ Run on Robot arm local network. """
    def __init__(self,
                 gpu_service_connector: ServiceConnector,
                 run_loops: int = 100,
                 img_hw: tuple = (256, 256),
                 conduct_actions_per_step: int = 1,
                 ):
        self.connector = gpu_service_connector
        self.run_loops = run_loops
        self.img_hw = img_hw
        self.conduct_actions_per_step = conduct_actions_per_step
        self.start_time = time.time()

        # Record
        self.log_time_delays = []

    def time_tick(self):
        self.start_time = time.time()

    def time_tok(self):
        time_delay = time.time() - self.start_time
        print(f"[DebugEvaluator] time cost: {time_delay * 1000.:.1f} ms")
        self.log_time_delays.append(time_delay)

    def show_time_delays(self):
        delays = np.array(self.log_time_delays) * 1000.  # second -> ms
        fps_lists = (1000. / (delays + 1e-6)).astype(np.int32)
        print(f"[DebugEvaluator] time delay: min={np.min(delays):.1f} ms, max={np.max(delays):.1f} ms, "
              f"avg={np.average(delays):.1f} ms; "
              f"min_fps={np.min(fps_lists)} FPS, max_fps={np.max(fps_lists)} FPS, "
              f"avg_fps={np.average(fps_lists)} FPS.")

    def run(self) -> float:
        cur_task_text = "DEBUG"
        self.connector.reset(cur_task_text)

        for i in range(self.run_loops):
            self.time_tick()

            if i % self.conduct_actions_per_step != 0:
                pass  # skip model inference
            else:
                cur_primary_rgb = np.random.randn(self.img_hw[0], self.img_hw[1], 3).astype(np.float32)
                cur_gripper_rgb = np.random.randn(self.img_hw[0], self.img_hw[1], 3).astype(np.float32)

                cur_primary_rgb = (cur_primary_rgb * 255).astype(np.uint8)
                cur_gripper_rgb = (cur_gripper_rgb * 255).astype(np.uint8)
                cur_joint_states = np.random.randn(6).astype(np.float32).tolist()
                actions = self.connector.step(
                    primary_rgb=cur_primary_rgb,
                    gripper_rgb=cur_gripper_rgb,
                    task_description=cur_task_text,
                    joint_states=cur_joint_states
                )
                assert actions.shape[1] == 6
                pass  # conduct actions in real-world environment

            time.sleep(0.01)  # Robot arm conducting delay
            self.time_tok()

        self.show_time_delays()

        return 0.



