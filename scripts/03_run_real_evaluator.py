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
]


def main(opts):
    robot_ip = '192.168.1.7'
    client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
    client.run()
    # Sanity check to see if we are connected
    print('Verifying the ROS target is connected?', client.is_connected)
    robot_client = RobotClient(client)

    if opts.port > 0:
        test_url = f"http://localhost:{opts.port}"
    elif opts.port == 0:
        test_url = "http://g7-debug.hkueai.org"
    else:
        raise Exception("Port number should be non-negative.")

    print("[Info] Using task:", task_instructions[opts.task])
    gpu_connector = ServiceConnector(base_url=test_url)
    evaluator = RealWorldEvaluator(
        gpu_service_connector=gpu_connector,
        robot=robot_client,
        cur_task_text=task_instructions[opts.task],
        run_loops=5000,
        img_hw=(480, 848),  # ori:(480, 848)
        resize_hw=(128, 160),  # ori:(128, 160)
        buffer_size=5,  # ori:1
        enable_auto_ae_wb=True,
        fps=30,
        speed_scale=1.,
        action_save_flag=opts.action_save_flag,
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
    python scripts/03_run_real_evaluator.py  \
        -p 6060
    """
    args = argparse.ArgumentParser("03 Run Real Evaluator")
    # debug: 6060
    # zhihao: 6260
    args.add_argument("-p", "--port", default=5880, type=int, help="Port number of GPU connects")
    args.add_argument("-t", "--task", default=-1, type=int, help="Index of task")
    args.add_argument("-a", "--action_save_flag", default=False, action="store_true", help="Action save flag")
    args = args.parse_args()
    main(args)
