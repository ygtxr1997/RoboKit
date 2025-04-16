import time

import roslibpy

from robokit.network.robot_client import RobotClient


robot_ip = '192.168.1.7'
client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
client.run()

# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)

rc = RobotClient(client)

time.sleep(1)

for _ in range(10000000):
    rc.get_current_frame_info()
    time.sleep(0.1)

print("OK!")
