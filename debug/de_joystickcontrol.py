import time

import roslibpy

from robokit.network.robot_client import RobotClient
from robokit.network.tele_control import PS5DualSenseIMUController


robot_ip = '192.168.1.7'
client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
client.run()

# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)

env_source_params = {
    "enable_auto_ae_wb": True,
}

env_light_params = {
    "enable_auto_ae_wb": False,
    "ae_wb_params": [{
        "camera_idx": -1,
        "exposure": 50,
    }],
}

rc = RobotClient(client)
controller = PS5DualSenseIMUController(
    robot=rc,
    saving_root="/home/geyuan/local_soft/TCL/collected_data_0513_table/",
    task_instruction="pick up the banana",
    saving_workers=6,
    **env_source_params
)

controller.start()

print("OK!")