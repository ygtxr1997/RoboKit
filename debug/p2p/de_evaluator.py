import roslibpy

from robokit.service.service_connector import ServiceConnector
from robokit.network.robot_evaluator import RealWorldEvaluator
from robokit.network.robot_client import RobotClient


robot_ip = '192.168.1.7'
client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
client.run()
# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)
robot_client = RobotClient(client)

# debug: 6060
# zhihao: 6260
gpu_connector = ServiceConnector(base_url="http://localhost:5880")
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
