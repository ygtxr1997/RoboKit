import argparse

import roslibpy

from robokit.service.service_connector import ServiceConnector
from robokit.network.robot_evaluator import RealWorldEvaluator
from robokit.network.robot_client import RobotClient


def main(opts):
    robot_ip = '192.168.1.7'
    client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
    client.run()
    # Sanity check to see if we are connected
    print('Verifying the ROS target is connected?', client.is_connected)
    robot_client = RobotClient(client)

    gpu_connector = ServiceConnector(base_url=f"http://localhost:{opts.port}")
    evaluator = RealWorldEvaluator(
        gpu_service_connector=gpu_connector,
        robot=robot_client,
        run_loops=5000,
        img_hw=(480, 848),  # (480, 848)
        enable_auto_ae_wb=True,
        fps=30,
    )  # TODO: better AE-WB setting
    try:
        evaluator.run()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("用户按下 Ctrl+C，程序中断。")
    finally:
        evaluator.stop()


if __name__ == "__main__":
    """
    Example usage:
    python scripts/01_preprocess_data.py  \
        -R "/home/geyuan/local_soft/TCL/collected_data_0507"  \
        --as_hdf5 "/home/geyuan/local_soft/TCL/hdf5/collected_data_0507.h5"
    """
    args = argparse.ArgumentParser("03 Run Real Evaluator")
    # debug: 6060
    # zhihao: 6260
    args.add_argument("-p", "--port", default=5880, type=int, help="Port number of GPU service")
    args = args.parse_args()
    main(args)
