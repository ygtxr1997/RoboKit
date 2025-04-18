#!/usr/bin/env python3

import os
import roslibpy
import time
# from inovopy.robot import InovoRobot
import roslibpy.actionlib

from robot_client import RobotClient
import pygame ## A library used to read joystick inputs

### Note: This was tested using the Logitech F310 gamepad ###

robot_ip = '192.168.1.7'
# bot = InovoRobot.default_iva(robot_ip)
# bot.gipper_activate()

try:
  client = roslibpy.Ros(host='192.168.1.7', port=9090) # Change host to the IP of the robot
  client.run()
except:
  print("Cannot connect to the robot, check your IP addess and network connection")
  exit()

# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)    # HW - If its not connected then we should terminate the progam here #AA - That's taken care of in the try loop above, this is only a sanity check that is adapted from previous examples

# Check 1: Topics
print("[DEBUG] Get topics:")
topics = client.get_topics()
for topic in topics:
    if "gripper" in topic:
        print(topic)
'''
/devices/robotiqd/gripper_state
/devices/robotiqd/gripper_command/result
/devices/robotiqd/gripper_command/feedback
/devices/robotiqd/gripper_command/status
/devices/robotiqd/gripper_command/goal
/devices/robotiqd/gripper_command/cancel
'''

# Check 2: Topic Message Type
print("[DEBUG] Get message type:")
topic_type = client.get_topic_type(topic='/devices/robotiqd/gripper_command/status')
'''
gripper_command/goal:
control_msgs/GripperCommandActionGoal
{'typedefs': [{'type': 'control_msgs/GripperCommandActionGoal', 'fieldnames': ['header', 'goal_id', 'goal'], 'fieldtypes': ['std_msgs/Header', 'actionlib_msgs/GoalID', 'control_msgs/GripperCommandGoal'], 'fieldarraylen': [-1, -1, -1], 'examples': ['{}', '{}', '{}'], 'constnames': [], 'constvalues': []}, {'type': 'std_msgs/Header', 'fieldnames': ['seq', 'stamp', 'frame_id'], 'fieldtypes': ['uint32', 'time', 'string'], 'fieldarraylen': [-1, -1, -1], 'examples': ['0', '{}', ''], 'constnames': [], 'constvalues': []}, {'type': 'time', 'fieldnames': ['secs', 'nsecs'], 'fieldtypes': ['int32', 'int32'], 'fieldarraylen': [-1, -1], 'examples': ['0', '0'], 'constnames': [], 'constvalues': []}, {'type': 'actionlib_msgs/GoalID', 'fieldnames': ['stamp', 'id'], 'fieldtypes': ['time', 'string'], 'fieldarraylen': [-1, -1], 'examples': ['{}', ''], 'constnames': [], 'constvalues': []}, {'type': 'control_msgs/GripperCommandGoal', 'fieldnames': ['command'], 'fieldtypes': ['control_msgs/GripperCommand'], 'fieldarraylen': [-1], 'examples': ['{}'], 'constnames': [], 'constvalues': []}, {'type': 'control_msgs/GripperCommand', 'fieldnames': ['position', 'max_effort'], 'fieldtypes': ['float64', 'float64'], 'fieldarraylen': [-1, -1], 'examples': ['0.0', '0.0'], 'constnames': [], 'constvalues': []}]}

gripper_state:
gripper_msgs/GripperState
{'typedefs': [{'type': 'gripper_msgs/GripperState', 'fieldnames': ['header', 'position', 'target_position', 'current', 'stalled', 'reached_goal', 'activated', 'safety_switch_triggered'], 'fieldtypes': ['std_msgs/Header', 'float64', 'float64', 'float64', 'bool', 'bool', 'bool', 'bool'], 'fieldarraylen': [-1, -1, -1, -1, -1, -1, -1, -1], 'examples': ['{}', '0.0', '0.0', '0.0', 'False', 'False', 'False', 'False'], 'constnames': [], 'constvalues': []}, {'type': 'std_msgs/Header', 'fieldnames': ['seq', 'stamp', 'frame_id'], 'fieldtypes': ['uint32', 'time', 'string'], 'fieldarraylen': [-1, -1, -1], 'examples': ['0', '{}', ''], 'constnames': [], 'constvalues': []}, {'type': 'time', 'fieldnames': ['secs', 'nsecs'], 'fieldtypes': ['int32', 'int32'], 'fieldarraylen': [-1, -1], 'examples': ['0', '0'], 'constnames': [], 'constvalues': []}]}

gripper_command/status:
actionlib_msgs/GoalStatusArray
{'typedefs': [{'type': 'actionlib_msgs/GoalStatusArray', 'fieldnames': ['header', 'status_list'], 'fieldtypes': ['std_msgs/Header', 'actionlib_msgs/GoalStatus'], 'fieldarraylen': [-1, 0], 'examples': ['{}', '[]'], 'constnames': [], 'constvalues': []}, {'type': 'std_msgs/Header', 'fieldnames': ['seq', 'stamp', 'frame_id'], 'fieldtypes': ['uint32', 'time', 'string'], 'fieldarraylen': [-1, -1, -1], 'examples': ['0', '{}', ''], 'constnames': [], 'constvalues': []}, {'type': 'time', 'fieldnames': ['secs', 'nsecs'], 'fieldtypes': ['int32', 'int32'], 'fieldarraylen': [-1, -1], 'examples': ['0', '0'], 'constnames': [], 'constvalues': []}, {'type': 'actionlib_msgs/GoalStatus', 'fieldnames': ['goal_id', 'status', 'text'], 'fieldtypes': ['actionlib_msgs/GoalID', 'uint8', 'string'], 'fieldarraylen': [-1, -1, -1], 'examples': ['{}', '0', ''], 'constnames': ['ABORTED', 'ACTIVE', 'LOST', 'PENDING', 'PREEMPTED', 'PREEMPTING', 'RECALLED', 'RECALLING', 'REJECTED', 'SUCCEEDED'], 'constvalues': ['4', '1', '9', '0', '2', '6', '8', '7', '5', '3']}, {'type': 'actionlib_msgs/GoalID', 'fieldnames': ['stamp', 'id'], 'fieldtypes': ['time', 'string'], 'fieldarraylen': [-1, -1], 'examples': ['{}', ''], 'constnames': [], 'constvalues': []}]}
'''
print(topic_type)

# Check 3: Get Message Details
print("[DEBUG] Get message details:")
message_info = client.get_message_details(message_type=topic_type)
print(message_info)

# Check 4: Try Publisher
print("[DEBUG] Try publisher:")
# publisher = roslibpy.Topic(
#     client, '/devices/robotiqd/gripper_command/status', 'actionlib_msgs/GoalStatusArray'
# )
# publisher.publish(roslibpy.Message({
#     "status_list": [{
#         "goal_id": {
#             "id": '/devices/robotiqd/robotiqd-12-1743413571.694867401',
#         },
#         "status": 1,
#         "text": "ACTIVE",
#     }]
# }))

# publisher = roslibpy.Topic(
#     client, '/devices/robotiqd/gripper_command/goal', 'control_msgs/GripperCommandActionGoal'
# )
# publisher.publish(roslibpy.Message({
#     "goal": {
#         "command": {
#             "position": 1.0,
#         }
#     }
# }))

# publisher = roslibpy.Topic(
#     client, "/devices/robotiqd/grip/goal", "gripper_msgs/GripperBasicCommandActionGoal"
# )
# publisher.publish(roslibpy.Message({
#     "goal": {
#         "action": 0  # 0:GRIP, 1:RELEASE, 2:TOGGLE
#     }
# }))

# publisher = roslibpy.Topic(client, '/default_move_group/cartesian_jog', 'commander_msgs/CartesianJogDemand')
# publisher.publish(roslibpy.Message({"twist": {"linear": {
#     'x': 0.05,
# }}}))


# Check 5: Try Listener
print("[DEBUG] Try listener:")
listener = roslibpy.Topic(
    client, '/devices/robotiqd/gripper_state', 'gripper_msgs/GripperState'
)
# Heard talking: {'header': {'seq': 13415, 'stamp': {'secs': 1743408753, 'nsecs': 720458513}, 'frame_id': ''}, 'position': 0.8980392156862745, 'target_position': 1.0, 'current': 0.0, 'stalled': False, 'reached_goal': False, 'activated': False, 'safety_switch_triggered': False}
# listener = roslibpy.Topic(
#     client, '/devices/robotiqd/gripper_command/status', 'actionlib_msgs/GoalStatusArray'
# )
# listener.subscribe(lambda message: print(f'Heard talking: {message}'))


rc = RobotClient(client)
coords ={'x': 0.0,
         'y': 0.0,
         'z': 0.0}
for i in range(30):
    rc.gripper_set_pub(client, 1.)  # Init
    time.sleep(0.1)
for i in range(100):
    rc.gripper_set_pub(client, 0.)  # Init
    time.sleep(0.1)

exit()


# try:
#     while True:
#         # publisher.publish(roslibpy.Message({
#         #     "goal_id": {
#         #         "id": "RECALLING"
#         #     },
#         #     "goal": {
#         #         "command": {
#         #             "position": 10,
#         #         }
#         #     }
#         # }))
#         pass
#         time.sleep(0.02)
# except KeyboardInterrupt:
#     client.terminate()
# exit()


def getJoy(axis, joystick): # axis is in reference to the joystick motion
    demand = joystick.get_axis(axis)
    if demand < 0.005 and demand > -0.005: demand = 0 # Applying dead band to avoid drift when joystick is released

    return demand*0.1


try:
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    print("joysticks:", len(joysticks))
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    maxC = 0
    kill = False
    while not kill:
        for event in pygame.event.get():
            time.sleep(0)
        #get events from the queue

        coords['x'] = -getJoy(1, joystick)
        coords['y'] = getJoy(0, joystick)
        coords['z'] = getJoy(3, joystick)

        # print(coords)
        # exit()

        buttons = joystick.get_numbuttons()
        # for i in range(buttons):
        #     button = joystick.get_button(i)
            # print(f"Button {i:>2} value: {button}")
        # os.system("clear")

        if joystick.get_button(1): # read B to kill the program
            kill = True
        if joystick.get_button(7): # read RB
            rc.linear_jog_pub(client, coords)
        if joystick.get_button(6): # read LB
            coords = {
                'x': -coords['y'] * 3,
                'y': -coords['x'] * 3,
                'z': -coords['z'] * 3,
            }
            rc.ang_jog_pub(client, coords)
        if not (joystick.get_button(7) or joystick.get_button(6)): # if neither RB or LB are pressed set to zero to avoid drift
            coords['x'] = 0
            coords['y'] = 0
            coords['z'] = 0
            rc.linear_jog_pub(client, coords)
            rc.ang_jog_pub(client, coords)
        axis_rt = joystick.get_axis(4)  # read RT, init: 0, 1:close, -1:open
        if axis_rt <= 0:
            axis_rt = 0.  # 0:open, 1:close
        # if axis_rt >= 0.3:
        #     pass
        #     rc.gripper_set_pub(client, 'close')  # send "close"
        # else:
        #     pass
        #     rc.gripper_set_pub(client, 'open')  # send "open"
        rc.gripper_set_pub(client, axis_rt)
        # rc.gripper_set_pub(client, 1.)

        pygame.time.Clock().tick(60) # setting the frame rate (FPS/Hz)

    print("Program Killed") # Printed after exiting the loop by B press
    pygame.quit()

except Exception as  e:
    print('Failed to upload to ftp: '+ str(e))
time.sleep(0.5) 


# Clean up the connection to the robot
client.terminate()