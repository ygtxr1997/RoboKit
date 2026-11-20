import argparse
from pathlib import Path
import traceback
import sys

import roslibpy
import yaml

from robokit.connects.service_connector import ServiceConnector
from robokit.robots.robot_evaluator import RealWorldEvaluator
from robokit.robots.robot_client_inovo import RobotClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/eval/policy_tcp_force.yaml"
TASK_CONFIG = REPO_ROOT / "meta/task_id_to_instructions.yaml"


def _load_yaml(path: Path) -> dict:
    """Load one YAML mapping used by the real evaluator."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _load_task_instruction(task_id: int) -> str:
    task_map = _load_yaml(TASK_CONFIG)
    normalized = {int(key): value for key, value in task_map.items()}
    if task_id not in normalized:
        raise KeyError(f"Unknown task id {task_id}; available ids: {sorted(normalized)}")
    return str(normalized[task_id])


def main(opts):
    config = _load_yaml(Path(opts.config).expanduser().resolve())
    camera_config = config["camera"]
    policy_config = config["policy"]
    control_config = config["control"]
    recording_config = config["recording"]

    task_id = 0 if opts.task is None else opts.task
    task_text = _load_task_instruction(task_id)
    resize_hw = camera_config.get("resize_hw")
    resize_hw = None if resize_hw is None else tuple(resize_hw)
    cache_length = int(policy_config["cache_length"])
    if cache_length <= 0:
        raise ValueError("policy.cache_length must be positive")
    robot_ip = '192.168.1.7'
    client = roslibpy.Ros(host=robot_ip, port=9090)  # Change host to the IP of the robot
    client.run()
    print('Verifying the ROS target is connected?', client.is_connected)
    robot_client = RobotClient(client)

    if opts.port > 1000:
        test_url = f"http://localhost:{opts.port}"
    elif opts.port == 0:
        test_url = f"http://{opts.server}-debug.hkueai.org"
    elif opts.port < 10:
        test_url = f"http://{opts.server}-debug{opts.port}.hkueai.org"
    else:
        raise Exception("Port number should be non-negative.")
    print("[Info] Connecting to GPU service at:", test_url)
    print("[Info] Using task:", task_text)

    gripper_sticky = (opts.gripper_sticky if opts.gripper_sticky is not None
                      else control_config["gripper_sticky"])
    continuous_gripper = (opts.continuous_gripper if opts.continuous_gripper is not None
                          else control_config["continuous_gripper"])
    gpu_connector = ServiceConnector(
        base_url=test_url,
        max_cache_actions=cache_length,
        close_online=not opts.open_online,
        num_camera_views=2,
    )
    evaluator = RealWorldEvaluator(
        gpu_service_connector=gpu_connector,
        robot=robot_client,
        cur_task_text=task_text,
        task_id=task_id,
        run_loops=6000,
        img_hw=(480, 848),
        resize_hw=resize_hw,
        buffer_size=cache_length,
        enable_auto_ae_wb=True,
        camera_skipping=opts.camera_skipping,
        fps=30,
        speed_scale=1.,
        gripper_sticky=gripper_sticky,
        continuous_gripper=continuous_gripper,
        action_save_flag=opts.action_save_flag,
        image_save_fn=recording_config["image_save_fn"],
        go_home_at_begin=not opts.no_home,
        open_gripper_at_begin=not opts.no_open,
        reset_remote_api_at_begin=not opts.no_reset,
        sleep_frames_after_send=opts.sleep_frames,
        stop_during_inference=not opts.no_stop_during_inference,
        state_fields=policy_config["state_fields"],
        expected_state_dim=policy_config.get("expected_state_dim"),
        force_fz_limit=control_config["force_fz_limit"],
        record_policy=opts.record_policy,
    )
    reason = "completed"
    try:
        evaluator.run()
    except Exception:
        reason = "error"
        traceback.print_exc()
    except KeyboardInterrupt:
        reason = "interrupted"
        print("用户按下 Ctrl+C，程序中断。")
    finally:
        try:
            try:
                evaluator.stop_recording_motion()
            except Exception:
                reason = 'error'
                traceback.print_exc()
            if gpu_connector.recording is not None and evaluator.recording_stop is not None:
                result, note = 'unknown', ''
                if sys.stdin.isatty() and reason != 'error':
                    try:
                        answer = input('Result [s=success / f=failure / Enter=unknown]: ').strip().lower()
                        result = {'s': 'success', 'f': 'failure', 'success': 'success',
                                  'failure': 'failure'}.get(answer, 'unknown')
                        note = input('Note (optional): ').strip()
                    except (EOFError, KeyboardInterrupt):
                        pass
                print(gpu_connector.recording.finish(evaluator.recording_stop,
                    result=result, note=note, reason=reason))
        finally:
            evaluator.stop()


if __name__ == "__main__":
    args = argparse.ArgumentParser("03 Run Real Evaluator")
    args.add_argument("--record-policy", action="store_true", help="Record timestamped policy views to the GPU HDF5 service")
    args.add_argument("--config", default=str(DEFAULT_CONFIG), help="Policy YAML configuration")
    args.add_argument("-p", "--port", default=5880, type=int, help="Port number of GPU connects")
    args.add_argument("-s", "--server", default="g7", type=str, help="Server name in remote GPU servers.")
    args.add_argument("-t", "--task", default=None, type=int,
                      help="Task id from meta/task_id_to_instructions.yaml; also used by the optional task_id state field (default: 0)")
    args.add_argument("-a", "--action_save_flag", default=False, action="store_true", help="Action save flag")
    args.add_argument("--no_home", action="store_true", help="No home at the beginning")
    args.add_argument("--no_open", action="store_true", help="No opening gripper at beginning")
    args.add_argument("--no_reset", action="store_true", help="Not reset remote GPU model at the beginning")
    args.add_argument("--open_online", action="store_true", help="Open online update (set stage_flag=1)")
    args.add_argument("--sleep_frames", default=1., type=float,
                      help="Control periods to wait after the final action of a chunk (or run), regardless of whether jog is stopped")
    args.add_argument("--no_stop_during_inference", action="store_true",
                      help="Skip zero-velocity jog before GPU requests; keep chunk-end waiting and stop on run completion or exit")
    args.add_argument("-gs", "--gripper_sticky", default=None, type=int,
                      help="Override YAML gripper sticky level")
    args.add_argument("--continuous_gripper", default=None, action=argparse.BooleanOptionalAction,
                      help="Override YAML continuous gripper setting")
    args.add_argument("-cs", "--camera_skipping", default=100, type=int, help="Skip camera frames to stabilize.")
    main(args.parse_args())
