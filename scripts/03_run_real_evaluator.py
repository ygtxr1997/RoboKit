import argparse
import traceback

import roslibpy

from robokit.connects.service_connector import ServiceConnector
from robokit.robots.robot_evaluator import RealWorldEvaluator
from robokit.robots.robot_client_inovo import RobotClient


task_instructions = [
    # [0] Banana
    "pick up the banana",
    # [1] Pot
    "put the egg into the pot, then move the pot onto the stove",
    # [2] Pepper
    "orient the cup upright on its base and insert the chili pepper vertically into it.",
    # [3] Coffee
    "use a spoon to scoop one spoonful of coffee beans from the source cup, "
    "then pour the beans into the target cup.",
    # [4] Spoon
    "Pick up the spoon in the cup, and then place the spoon into a bowl.",
    # [5] Sweep
    "Pick up the broom and sweep the coffee beans on the table into the dustpan.",
    # [6] Eggs
    "Pick up two eggs from a plate, and then place the eggs into a egg carton.",
    # [7] Pour water
    "Pick up the measuring cup and pour the water into a glass.",
    # [8] Wipe
    "Wiping the writing on the blackboard.",
    # [9] Red Pepper
    "Pick up the red pepper on the box, and then place it into a cup.",
    # [10] Carrot
    "Pick up the carrot from a plate, and then place it into a bowl.",
    # [11] Screw Bulb
    "Screw the bulb into the light socket, and turn on the light.",
    # [12] Screw Bulb Turn Off
    "Screw the bulb into the light socket, and turn off the light after the bulb is on.",
    # [13] Ethernet
    "Disconnect the ethernet cable from the `uplink` port, then insert it into port `1`.",
]


def main(opts):
    robot_ip = '192.168.1.7'
    client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
    client.run()
    # Sanity check to see if we are connected
    print('Verifying the ROS target is connected?', client.is_connected)
    robot_client = RobotClient(client)

    if opts.port > 1000:
        test_url = f"http://localhost:{opts.port}"
    elif opts.port == 0:
        test_url = "http://g7-debug.hkueai.org"
    elif opts.port < 10:
        test_url = f"http://g7-debug{opts.port}.hkueai.org"
    else:
        raise Exception("Port number should be non-negative.")
    print("[Info] Connecting to GPU service at:", test_url)

    print("[Info] Using task:", task_instructions[opts.task])
    gpu_connector = ServiceConnector(
        base_url=test_url,
        max_cache_actions=24,  # ori:24
        close_online=not opts.open_online,
        num_camera_views=2,
    )
    evaluator = RealWorldEvaluator(
        gpu_service_connector=gpu_connector,
        robot=robot_client,
        cur_task_text=task_instructions[opts.task],
        run_loops=6000,  # ori:5000
        img_hw=(480, 848),  # ori:(480, 848)
        resize_hw=(240, 320),  # k1:(16*7, 16*10), k8:(16*10, 16*15), dp:(240, 320)
        buffer_size=24,  # ori:1
        enable_auto_ae_wb=True,
        camera_skipping=opts.camera_skipping,
        fps=30,  # ori:30
        speed_scale=1.,  # ori:1.
        gripper_sticky=opts.gripper_sticky,
        continuous_gripper=opts.continuous_gripper,
        action_save_flag=opts.action_save_flag,
        go_home_at_begin=not opts.no_home,
        open_gripper_at_begin=not opts.no_open,
        reset_remote_api_at_begin=not opts.no_reset,
        sleep_frames_after_send=opts.sleep_frames,
    )  # TODO: better AE-WB setting
    try:
        evaluator.run()
    except Exception as e:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("用户按下 Ctrl+C，程序中断。")
    finally:
        evaluator.stop()


if __name__ == "__main__":
    """
    Example usage:
    [Local infer]:
    python scripts/03_run_real_evaluator.py  \
        -p 5880
    [Remote infer]:
    python scripts/03_run_real_evaluator.py  \
        -p 0
    """
    args = argparse.ArgumentParser("03 Run Real Evaluator")
    # debug: 6060
    # zhihao: 6260
    args.add_argument("-p", "--port", default=5880, type=int, help="Port number of GPU connects")
    args.add_argument("-t", "--task", default=-1, type=int, help="Index of task")
    args.add_argument("-a", "--action_save_flag", default=False, action="store_true", help="Action save flag")
    args.add_argument("-k", "--future_skip", default=1, type=int, help="Skip for future frames (not supported yet)")
    args.add_argument("--no_home", action="store_true", help="No home at the beginning")
    args.add_argument("--no_open", action="store_true", help="No opening gripper at beginning")
    args.add_argument("--no_reset", action="store_true", help="Not reset remote GPU model at the beginning")
    args.add_argument("--open_online", action="store_true", help="Open online update (set stage_flag=1)")
    args.add_argument("--sleep_frames", default=1., type=float, help="Scale for sleep time after send")
    args.add_argument("-gs", "--gripper_sticky", default=0, type=int, help="Gripper sticky level (0: non-sticky, 80: super-sticky)")
    args.add_argument("--continuous_gripper", action="store_true",
                      help="Execute continuous closure targets (0 open, 1 closed); ignores gripper_sticky")
    args.add_argument("-cs", "--camera_skipping", default=100, type=int, help="Skip camera frames to stabilize.")
    args = args.parse_args()
    main(args)
