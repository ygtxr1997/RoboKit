import time

import roslibpy

from robokit.network.robot_client import RobotClient
from robokit.network.tele_control import SwitchProController, SwitchProIMUController


robot_ip = '192.168.1.7'
client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
client.run()

# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)

rc = RobotClient(client)
controller = SwitchProIMUController(
    robot=rc,
    saving_root="/home/geyuan/local_soft/TCL/collected_data_0425_tmp/",
    enable_auto_ae_wb=False,
)

controller.start()

print("OK!")