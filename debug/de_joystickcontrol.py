import time

import roslibpy

from robokit.network import RobotClient
from robokit.network import SwitchProController


robot_ip = '192.168.1.7'
client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
client.run()

# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)

rc = RobotClient(client)
controller = SwitchProController(robot=rc)

controller.start()

print("OK!")