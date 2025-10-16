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
        "exposure": 30,
    }],
}

### Params for data collector ###
''' Pot '''
saving_root = "/home/geyuan/local_soft/TCL/0627_pot_source/"
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_light_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_object_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_table_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_distractor_rand/"
task_instruction = "put the egg into the pot, then move the pot onto the stove"
''' Pepper '''
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_source/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_light_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_object_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_table_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_distractor_rand/"
# task_instruction = "orient the cup upright on its base and insert the chili pepper vertically into it."
''' Coffee '''
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_source/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_light_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_object_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_table_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_distractor_rand/"
# task_instruction = "use a spoon to scoop one spoonful of coffee beans from the source cup, then pour the beans into the target cup."

env_params = env_source_params

rc = RobotClient(client)
controller = PS5DualSenseIMUController(
    robot=rc,
    saving_root=saving_root,
    task_instruction=task_instruction,
    saving_workers=6,
    action_fps=30,  # ori:30, rand:60
    **env_params
)

try:
    controller.start()
except Exception as e:
    print(e)
except KeyboardInterrupt:
    print("用户按下 Ctrl+C，程序中断。")
finally:
    controller.stop()

print("OK!")