from __future__ import print_function

import time
from typing import List, Union, Dict
from enum import Enum
import pprint
import math
import copy

from scipy.spatial.transform import Rotation

import roslibpy
import roslibpy.actionlib


class GripperState(Enum):
    INIT = 0
    REACHED_OPEN = 1
    MOVING_TO_OPEN = 2
    REACHED_CLOSE = 3
    MOVING_TO_CLOSE = 4

    # STATE_ZERO = [0, 1, 2]  # open
    # STATE_ONE = [3, 4]  # close
    STATE_ZERO = [INIT, REACHED_OPEN, MOVING_TO_OPEN]  # open
    STATE_ONE = [REACHED_CLOSE, MOVING_TO_CLOSE]  # close


class RobotClient:
    SAFETY_CIRCUIT_OPEN = 0
    SAFETY_CIRCUIT_CLOSED = 1

    def __init__(self, ros):
        self._ros = ros

        # SUBSCRIBING TO GRIPPER STATUS
        self.gripper_listener = roslibpy.Topic(
            self._ros, '/devices/robotiqd/gripper_state', 'gripper_msgs/GripperState'
        )
        self.gripper_listener.subscribe(self.gripper_status)
        self.gripper = {
            'reached_goal': True,
            'position': 1.,
            'state': GripperState.INIT,  # managed by ourselves
            'message': 0.,  # 0:open, 1:close
        }

        # SUBSCRIBING to DEBUG info
        self.debug_listener = roslibpy.Topic(
            self._ros, "/default_move_group/move/result", 'commander_msgs/MotionActionResult'
        )
        def debug_printer(message):
            print(message)
        self.debug_listener.subscribe(debug_printer)

        # SUBSCRIBING TO TCP SPEED ON ROS
        self.tcp_speed_client = roslibpy.Topic(self._ros, '/default_move_group/tcp_speed',
                                               'commander_msgs/SpeedStamped')
        self.tcp_speed_client.subscribe(self.tcp_speed)
        self.Speed = {'Lin': 0,  # Dictionary of Speeds
                      'Ang': 0}

        # SUBSCRIBING TO TCP POSE ON ROS
        self.tcp_pose_client = roslibpy.Topic(self._ros, '/default_move_group/tcp_pose', 'geometry_msgs/PoseStamped')
        self.tcp_pose_client.subscribe(self.tcp_pose)
        self.Pose = {'Coords': {'x': 0, 'y': 0, 'z': 0},  # Dictionary of Poses
                     'Ori': {'x': 0, 'y': 0, 'z': 0, 'w': 0}}

        # SUBSCRIBING TO JOINT STATES
        self.joint_states_client = roslibpy.Topic(self._ros, '/robot/joint_states', 'sensor_msgs/JointState')
        self.joint_states_client.subscribe(self.joint_states)
        self.State = {'pos': [],  # A dictionary of array states
                      'vel': [],
                      'eff': []}

        self.power_state_client = roslibpy.Topic(self._ros, '/psu/status', 'psu_msgs/Status')
        self.power_state_client.subscribe(self.power_state)
        self.power = {'voltage': 0.0,
                      'current': 0.0,
                      'state': False}

        self.arm_ready_client = roslibpy.Topic(self._ros, '/robot/robot_state', 'arm_msgs/RobotState')
        self.arm_ready_client.subscribe(self.arm_ready)
        self.arm = {'driver_active': False,
                    'power': False}

        self.estop_state_client = roslibpy.Topic(self._ros, '/psu/estop/state', 'psu_msgs/SafetyCircuitState')
        self.estop_state_client.subscribe(self.estop_state)
        self.estop = {'active': False,
                      'circuit': False}

        self.safe_stop_state_client = roslibpy.Topic(self._ros, '/psu/safe_stop/state', 'psu_msgs/SafetyCircuitState')
        self.safe_stop_state_client.subscribe(self.safety_stop_state)
        self.safe_stop = {'active': False,
                          'circuit': False}

        #### Services ####

        ## Arm and PSU power and activation ##
        self.safe_stop_reset_service = roslibpy.Service(self._ros, '/psu/safe_stop/reset', 'std_srvs/Trigger')
        self.estop_reset_service = roslibpy.Service(self._ros, '/psu/estop/reset', 'std_srvs/Trigger')

        self.power_on_service = roslibpy.Service(self._ros, '/psu/enable', 'std_srvs/Trigger')
        self.power_off_service = roslibpy.Service(self._ros, '/psu/disable', 'std_srvs/Trigger')

        self.arm_on_service = roslibpy.Service(self._ros, '/robot/enable', 'std_srvs/Trigger')
        self.arm_off_service = roslibpy.Service(self._ros, '/robot/disable', 'std_srvs/Trigger')

        self.gripper_service = roslibpy.Service(self._ros, '/devices/robotiqd/activate', 'std_srvs/Trigger')

        #### Motion ####
        self.joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        self.message_linear_jog: dict = {'x': 0, 'y':0, 'z': 0}
        self.message_angular_jog: dict = {'x': 0, 'y':0, 'z': 0}
        self.message_zero_jog: dict = {'x': 0, 'y': 0, 'z': 0}

    #### TOPIC FUNCTIONS ####
    def gripper_status(self, message):
        self.gripper['reached_goal'] = bool(message['reached_goal'])
        self.gripper['position'] = float(message['position'])

    def tcp_speed(self, message):
        self.Speed['Lin'] = message['speed']['linear']
        self.Speed['Ang'] = message['speed']['angular']

    def get_tcp_linear_speed(self):
        return self.Speed['Lin']

    def get_tcp_angular_speed(self):
        return self.Speed['Ang']

    def tcp_pose(self, message):
        # Setting the coordinates
        self.Pose['Coords']['x'] = message['pose']['position']['x']
        self.Pose['Coords']['y'] = message['pose']['position']['y']
        self.Pose['Coords']['z'] = message['pose']['position']['z']

        # Setting the orientations
        self.Pose['Ori']['x'] = message['pose']['orientation']['x']
        self.Pose['Ori']['y'] = message['pose']['orientation']['y']
        self.Pose['Ori']['z'] = message['pose']['orientation']['z']
        self.Pose['Ori']['w'] = message['pose']['orientation']['w']

    def get_tcp_coordinates(self):
        """ TCP xyz (m) """
        return self.Pose['Coords']

    def get_tcp_orientation(self, out_type: str = 'euler'):
        """ TCP orientation
        Supported out_type: ['euler', 'quaternion', 'euler_degree']
        """
        if out_type == 'quaternion':
            return self.Pose['Ori']  # (quaternion)
        elif 'euler' in out_type:
            quaternion = [self.Pose['Ori']['x'], self.Pose['Ori']['y'], self.Pose['Ori']['z'], self.Pose['Ori']['w']]
            rotation = Rotation.from_quat(quaternion)
            is_degree = 'degree' in out_type
            euler_angles = rotation.as_euler('xyz', degrees=is_degree)
            return euler_angles  # (euler)
        else:
            raise NotImplementedError

    def joint_states(self, message):
        jointNum = len(message['position'])  # Number of joints from the ROS dictionary

        # Giving the arrays the correct size according to the number of joints
        self.State['pos'] = [0] * jointNum
        self.State['vel'] = [0] * jointNum
        self.State['eff'] = [0] * jointNum

        # for each joint number, set the corresponding joint stat
        for x in range(jointNum):
            self.State['pos'][x] = message['position'][x]
            self.State['vel'][x] = message['velocity'][x]
            self.State['eff'][x] = message['effort'][x]

    def get_joint_angles(self, out_type='radius'):  # get_joint_angles()[angle_num] or get_JointAngles() to get an array of them
        if 'radius' in out_type:
            return self.State['pos']
        elif 'degree' in out_type:
            return [x * 180 / math.pi for x in self.State['pos']]

    def get_joint_velocity(self):
        return self.State['vel']

    def get_joint_effort(self):
        return self.State['eff']

    ## POWER AND ARM STATUS TOPICS
    def power_state(self, message):
        if message['state'] == "BUS_ON":
            self.power['state'] = True
        else:
            self.power['state'] = False
        self.power["voltage"] = message["voltage"]
        self.power["current"] = message["current"]

    def get_power_state(self):
        return self.power

    def arm_ready(self, message):
        if message['driver_state'] == 'Active':
            self.arm['driver_active'] = True
        else:
            self.arm['state'] = False
        self.arm['powered'] = message['drives_powered']

    def get_arm_power(self):
        return self.arm['powered']

    def get_arm_active(self):
        return self.arm['driver_active']

    def estop_state(self, message):
        self.estop['active'] = message['active']
        self.estop['circuit'] = message['circuit_complete']

    def get_estop_state(self):
        return self.estop

    def safety_stop_state(self, message):
        self.safe_stop['active'] = message['active']
        self.safe_stop['circuit'] = message['circuit_complete']

    def get_safety_stop_state(self):
        return self.safe_stop

    ##### SERVICES #####
    def safe_stop_reset(self):  ## MAKE SURE IT IS PHYSICALLY INACTIVE
        request = roslibpy.ServiceRequest()
        result = self.safe_stop_reset_service.call(request)
        if not result['success']:
            raise Exception(f"Unable to reset safety stop: {result['message']}")

    def estop_reset(self):
        request = roslibpy.ServiceRequest()
        result = self.estop_reset_service.call(request)
        if not result['success']:
            raise Exception(f"Unable to reset emergency stop: {result['message']}")

    def arm_power_on(self):
        request = roslibpy.ServiceRequest()
        result = self.power_on_service.call(request)
        print("[Service] Opening power...")
        time.sleep(5)
        if not result['success']:
            raise Exception(f"Unable to turn on power: {result['message']}")

    def arm_power_off(self):
        request = roslibpy.ServiceRequest()
        result = self.power_off_service.call(request)
        if not result['success']:
            raise Exception(f"Unable to turn off power: {result['message']}")

    def robot_arm_enable(self):
        request = roslibpy.ServiceRequest()
        result = self.arm_on_service.call(request)
        time.sleep(1)
        if not result['success']:
            raise Exception(f"Unable to enable the arm: {result['message']}")

    def robot_arm_disable(self):
        request = roslibpy.ServiceRequest()
        result = self.arm_off_service.call(request)
        if not result['success']:
            raise Exception(f"Unable to disable the arm: {result['message']}")

    def robot_gripper_enable(self):
        request = roslibpy.ServiceRequest()
        result = self.gripper_service.call(request)
        print("[Service] Gripper activated!")
        if not result['success']:
            raise Exception(f"Unable to activate gripper: {result['message']}")

    def set_power(self, bool):
        if bool == True:
            self.arm_power_on()
        elif bool == False:
            self.arm_power_off()
        else:
            raise Exception(f"Unable to set state")

    def set_arm(self, bool):
        if bool == True:
            self.robot_arm_enable()
        elif bool == False:
            self.robot_arm_disable()
        else:
            raise Exception(f"Unable to set state")

    #### CARTESIAN JOG MOTION ####
    ## the functions take in the client that connects to the robot and a message to be sent
    ## the message must be of type dictionary with elements of 'x', 'y' and 'z'
    ## the cartesian jog allows for offsetting the arm by the input to each coordinate
    ## linear jog offsets the x, y, and z linearly
    ## angular jog causes a rotation of the TCP
    def linear_jog_pub(self, message):
        client = self._ros
        self.message_linear_jog = message
        self.message_angular_jog = self.message_zero_jog  # set angular as zero
        publisher = roslibpy.Topic(
            client, '/default_move_group/cartesian_jog',
            'commander_msgs/CartesianJogDemand')
        publisher.publish(roslibpy.Message({"twist": {"linear": message}}))

    def ang_jog_pub(self, message: dict):
        client = self._ros
        self.message_linear_jog = self.message_zero_jog  # set linear as zero
        self.message_angular_jog = message
        publisher = roslibpy.Topic(
            client, '/default_move_group/cartesian_jog',
            'commander_msgs/CartesianJogDemand')
        publisher.publish(roslibpy.Message({"twist": {"angular": message}}))

    def lin_ang_jog_pub(self, linear_message: dict, angular_message: dict):
        client = self._ros
        self.message_linear_jog = linear_message
        self.message_angular_jog = angular_message
        publisher = roslibpy.Topic(
            client, '/default_move_group/cartesian_jog',
            'commander_msgs/CartesianJogDemand')
        publisher.publish(roslibpy.Message({"twist": {
            "linear": linear_message,
            "angular": angular_message
        }}))

    #### GRIPPER CONTROL ####
    def _send_gripper_action(self, action: int):
        # Send, 0:GRIP, 1:RELEASE, 2:TOGGLE
        client = self._ros
        publisher = roslibpy.Topic(
            client, "/devices/robotiqd/grip/goal",
            "gripper_msgs/GripperBasicCommandActionGoal"
        )
        publisher.publish(roslibpy.Message({
            "goal": {
                "action": action  # 0:GRIP, 1:RELEASE, 2:TOGGLE
            }
        }))

    def gripper_set_pub(self,
                        message: float):  # message:0-open, 1-close
        """ message: 0: To Open; 1: To Close """
        client = self._ros
        self.gripper['message'] = message

        DIST_EPS = 0.01
        POS_MAX = 0.99
        POS_MIN = 0.01
        position_goal = 1. - message  # distance:0-min, 1-max
        position_goal = max(min(position_goal, POS_MAX), POS_MIN)
        dist = abs(position_goal - float(self.gripper['position']))
        has_new_goal = bool(dist >= DIST_EPS)
        is_reached = bool(dist <= DIST_EPS)

        # print(
        #     f"[DEBUG]: has_new_goal={has_new_goal}, state={self.gripper['state']}; "
        #     f"reached={self.gripper['reached_goal']}, message={message}, "
        #     f"goal={position_goal}, position={self.gripper['position']}")

        if message <= 0.5:  # Open
            if self.gripper['state'] in [
                GripperState.MOVING_TO_OPEN]:
                # Need to check if reached
                if is_reached:
                    self.gripper['state'] = GripperState.REACHED_OPEN
            elif self.gripper['state'] in [
                GripperState.REACHED_OPEN
            ]:
                # Do nothing
                return
            else:
                assert self.gripper['state'] in [
                    GripperState.INIT,
                    GripperState.MOVING_TO_CLOSE,  # interrupted
                    GripperState.REACHED_CLOSE,
                ]
                self._send_gripper_action(action=1)
                self.gripper['state'] = GripperState.MOVING_TO_OPEN
        else:  # Close
            if self.gripper['state'] in [
                GripperState.MOVING_TO_CLOSE]:
                # Need to check if reached
                if is_reached:
                    self.gripper['state'] = GripperState.REACHED_CLOSE
            elif self.gripper['state'] in [
                GripperState.REACHED_CLOSE
            ]:
                # Do nothing
                return
            else:
                assert self.gripper['state'] in [
                    GripperState.INIT,
                    GripperState.MOVING_TO_OPEN,  # interrupted
                    GripperState.REACHED_OPEN,
                ]
                self._send_gripper_action(action=0)
                self.gripper['state'] = GripperState.MOVING_TO_CLOSE
        return

    def get_gripper_message(self):
        """ 0:Open; 1:Close """
        return self.gripper['message']  # binary, in {0,1}

    def get_gripper_opening_width(self):
        return self.gripper['position']  # ratio, in [0,1]

    #### MOTION CONTROL ####
    def joint_goal_send(self, positions: List[float],
                        velocities: List[float] = None,
                        seconds_from_start: float = 10,
                        timeout: float = 60,
                        ):
        if velocities is None:
            velocities = [0.02] * len(positions)
        trajectory_goal_settings = {  # Dictionary in case of trajectory ### WIP ###
            'positions': positions,
            'velocities': velocities,
            'time_from_start': {'secs': seconds_from_start}  # Default time
        }
        trajectory_goals = [trajectory_goal_settings]

        service = roslibpy.Service(self._ros, '/robot/switch_controller', 'inovo_driver/SwitchControllerGroup')
        request = roslibpy.ServiceRequest({'name': 'trajectory'})
        result = service.call(request)
        # print('Service response: {}'.format(result))

        # Setting up the client for the trajectory action
        action_client = roslibpy.actionlib.ActionClient(
            self._ros, '/robot/joint_trajectory_controller/follow_joint_trajectory',
            'control_msgs/FollowJointTrajectoryAction')

        message_ = {
            'trajectory': {  # Compiling the message of different dictionaries and arrays to be sent to the server
                'joint_names': self.joint_names,
                'points': trajectory_goals
        }}

        # print(message_)

        goal = roslibpy.actionlib.Goal(action_client, roslibpy.Message(message_))

        goal.on('[joint_goal_send] feedback:', lambda f: print(f))
        goal.send()

        print("[DEBUG] waiting for joint_goal...")
        result = goal.wait(timeout)
        action_client.dispose()
        print("[DEBUG] joint_goal finished!...")

    def tcp_goal_send(self, tcp_pos: dict, tcp_ori: dict,
                      timeout: float = 60):
        motion_goal_settings = {  # Dictionary in case of simple motion
            'pose': {
                'position': tcp_pos,  # {'x','y','z'}
                'orientation': tcp_ori ,  # {'x','y','z','w'}
            },
            'frame_id': 'world',  # Default frame
            'max_velocity': {'linear': 0.05,  ## Default velocities
                             'angular': 0.05},
            'max_joint_velocity': 0.05,
            'max_joint_acceleration': 0.05,
            'blend': {'linear': 0.0,
                      'angular': 0.0
                      }
        }
        goal_info = [motion_goal_settings]

        service = roslibpy.Service(
            self._ros, '/robot/switch_controller', 'inovo_driver/SwitchControllerGroup')
        request = roslibpy.ServiceRequest({'name': 'trajectory'})
        result = service.call(request)

        action_client = roslibpy.actionlib.ActionClient(
            self._ros, '/default_move_group/move',
            'commander_msgs/MotionAction')

        message_ = {
            'motion_sequence': goal_info}  ## Creating a dictionary that looks the same as the simple motion

        goal = roslibpy.actionlib.Goal(action_client, roslibpy.Message(message_))
        # goal.on('feedback', lambda f: print(f))
        goal.on("feedback", lambda f: print(f))
        ## Start the goal - this is where the robot will start moving!
        goal.send()

        print("[DEBUG] waiting for tcp_goal...")
        result = goal.wait(timeout)
        action_client.dispose()
        print("[DEBUG] tcp_goal finished!...")

    def joint_back_home(self):
        pi = math.pi
        home_joint_positions = [0, 0, pi / 2,
                                0, pi / 2, pi]
        self.joint_goal_send(home_joint_positions)

    def tcp_back_home(self):
        """ More safe comparing with joint_back_home """
        home_tcp_pos = {'x': 0.0, 'y': 0.4, 'z': 0.4567}
        home_tcp_ori = {'x': 1, 'y': 0, 'z': 0, 'w': 0}
        self.tcp_goal_send(home_tcp_pos, home_tcp_ori)

    #### Data related ####
    @staticmethod
    def beautify_print(d):
        """使用 pprint 库格式化打印字典"""
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(d)

    def get_current_frame_info(self, verbose=False):
        if self.gripper['state'].value in GripperState.STATE_ZERO.value:
            gripper_moving_to = 0
        else:
            assert self.gripper['state'].value in GripperState.STATE_ONE.value
            gripper_moving_to = 1

        data_dict = {
            "tcp_xyz_wrt_base": self.get_tcp_coordinates(),
            "tcp_ori_wrt_base": self.get_tcp_orientation(out_type='euler_radius'),
            "gripper_moving_to": gripper_moving_to,
            "jog_linear": self.message_linear_jog,
            "jog_angular": self.message_angular_jog,
            "gripper_width": self.gripper['position'],
            "joint_states": self.get_joint_angles(out_type='radius'),
        }

        if verbose:
            self.beautify_print(data_dict)

        return data_dict
