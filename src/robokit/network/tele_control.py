import os
import time
from datetime import datetime
import enum
from abc import ABC, abstractmethod
import pygame
import numpy as np

from .robot_client import RobotClient
from robokit.network.imu_control import RawIMUHandler
from robokit.data.data_handler import ForkedDataSaver
from robokit.data.realsense_handler import RealsenseHandler


class GameState(enum.Enum):
    INIT = 0
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


class BaseController:
    def __init__(self, robot: RobotClient, saving_root: str,
                 enable_auto_ae_wb: bool = True,
                 ):
        pygame.font.init()
        pygame.display.init()
        pygame.joystick.init()

        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("TCP Data Visualization")

        self.robot = robot
        self.linear_scale = 0.1
        self.angular_scale = 0.1
        self.fps = 30

        # Dynamic
        self.linear_xyz = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.angular_xyz = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.z_39 = 0.  # left-right axis of z, only used by angular jog move
        self.g = 0.  # 0:open, 1:close

        # Open gripper service
        self.robot.arm_power_on()
        self.robot.robot_arm_enable()
        self.robot.robot_gripper_enable()

        # Sanity check for gripper
        self.robot.gripper_set_pub(1)
        time.sleep(0.5)
        self.robot.gripper_set_pub(0)
        time.sleep(0.5)

        self.game_state = GameState.INIT

        # Cameras
        self.camera = RealsenseHandler(frame_rate=60)
        self.camera.set_ae_wb_auto(enable_auto_ae_wb)
        if not enable_auto_ae_wb:
            self.camera.set_ae_wb(exposure=50)
        self.data_manager = ForkedDataSaver(num_workers=6)
        self.need_saving = False
        self.saving_root = saving_root
        self.saving_dir: str = None
        self.saving_frame_idx: int = 0

    def get_datetime_str(self):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y_%m_%d-%H_%M_%S")
        return formatted_time

    def start(self):
        self.game_state = GameState.RUNNING
        toggle_flag = False
        last_game_time = pygame.time.get_ticks()

        while self.game_state != GameState.STOPPED:
            # Joystick FPS
            current_time = pygame.time.get_ticks()
            self.screen.fill((255, 255, 255))

            angular_release = False
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == self.get_pause_button() and not toggle_flag:
                        self.switch_pause()
                        toggle_flag = True  # 状态切换后标记为已切换
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button == self.get_pause_button():
                        toggle_flag = False  # 当按钮释放时允许再次切换
                    elif event.button == self.get_angular_button():
                        angular_release = True
                        self.on_angular_button_released()

            self.update_state()
            self.update_xyz()
            self.update_g()

            # DEBUG
            # print("[DEBUG] tcp position (m):", self.robot.get_tcp_coordinates())
            # print("[DEBUG] tcp linear_jog (m):", self.robot.message_linear_jog)
            #
            # print("[DEBUG] tcp orientation (radius):", self.robot.get_tcp_orientation('euler'))
            # print("[DEBUG] tcp orientation (360):", self.robot.get_tcp_orientation('euler_degree'))
            # print("[DEBUG] tcp angular_jog (radius):", self.robot.message_angular_jog)
            #
            # print("[DEBUG] tcp gripper_action (binary):", self.robot.get_gripper_message())
            # print("[DEBUG] gripper opening width (100%):", self.robot.get_gripper_opening_width())
            #
            # print("[DEBUG] arm_joint_states (radius):", self.robot.get_joint_angles())
            # print("[DEBUG] arm_joint_states (360):", self.robot.get_joint_angles('degree'))
            # print("===" * 20)

            start_y = 50
            self.debug_add_bar('tcp linear_jog x (m):', self.robot.message_linear_jog['x'], pos_y=start_y)
            start_y += 50
            self.debug_add_bar('tcp linear_jog y (m):', self.robot.message_linear_jog['y'], pos_y=start_y)
            start_y += 50
            self.debug_add_bar('tcp linear_jog z (m):', self.robot.message_linear_jog['z'], pos_y=start_y)

            # self.robot.get_current_frame_info()

            # Check Exit
            if self.is_exit_pressed():
                self.game_state = GameState.STOPPED

            if current_time - last_game_time >= 1000 / self.fps:
                # Robot FPS
                last_game_time = current_time

                if self.game_state == GameState.PAUSED:
                    pass
                else:
                    # Check Back Home
                    if self.is_back_pressed():
                        self.on_back_home()

                    # Check XYZ Move
                    zero_xyz = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    if self.is_linear_jog_pressed():
                        self.on_linear_jog()
                    elif self.is_angular_jog_pressed():
                        self.on_angular_jog()
                    else:
                        self.on_linear_jog(zero_xyz)
                        self.on_angular_jog(zero_xyz, z_39=0.)

                    # Check Gripper
                    self.on_gripper_move()

                    # Check Episode start end
                    if self.is_start_episode_pressed():
                        self.on_start_episode()
                    elif self.is_end_episode_pressed():
                        self.on_end_episode()

                # Save data
                if self.need_saving:
                    assert self.saving_dir is not None
                    os.makedirs(self.saving_dir, exist_ok=True)

                    start_time = pygame.time.get_ticks()
                    camera_data = self.camera.capture_frames()
                    # print("[DEBUG] Get camera data cost:", pygame.time.get_ticks() - start_time)
                    task_instruction = "pick up the banana"
                    start_time = pygame.time.get_ticks()
                    robot_data = self.robot.get_current_frame_info()
                    # print("[DEBUG] Get robot data cost:", pygame.time.get_ticks() - start_time)

                    # Note: if not set linear_jog to zero, will save wrong data
                    zero_xyz = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    if self.is_linear_jog_pressed():
                        robot_data['jog_angular'] = zero_xyz
                    elif self.is_angular_jog_pressed():
                        robot_data['jog_linear'] = zero_xyz
                    else:
                        robot_data['jog_angular'] = zero_xyz
                        robot_data['jog_linear'] = zero_xyz

                    frame_actions = np.array([
                        robot_data['jog_linear']['x'],
                        robot_data['jog_linear']['y'],
                        robot_data['jog_linear']['z'],
                        robot_data['jog_angular']['x'],
                        robot_data['jog_angular']['y'],
                        robot_data['jog_angular']['z'],
                        robot_data['gripper_moving_to'],
                    ])
                    frame_rel_actions = frame_actions  # TODO: using same action space with absolute actions
                    frame_robot_obs = np.array([
                        robot_data['tcp_xyz_wrt_base']['x'],
                        robot_data['tcp_xyz_wrt_base']['y'],
                        robot_data['tcp_xyz_wrt_base']['z'],
                        robot_data['tcp_ori_wrt_base'][0],
                        robot_data['tcp_ori_wrt_base'][1],
                        robot_data['tcp_ori_wrt_base'][2],
                        robot_data['gripper_width'],
                        robot_data['joint_states'][0],
                        robot_data['joint_states'][1],
                        robot_data['joint_states'][2],
                        robot_data['joint_states'][3],
                        robot_data['joint_states'][4],
                        robot_data['joint_states'][5],
                        robot_data['gripper_moving_to'],
                    ])

                    frame_data_dict = {
                        "primary_rgb": camera_data['color2'],
                        "gripper_rgb": camera_data['color1'],
                        "primary_depth": camera_data['depth2'],
                        "gripper_depth": camera_data['depth1'],
                        "language_text": task_instruction,
                        "actions": frame_actions,
                        "rel_actions": frame_rel_actions,
                        "robot_obs": frame_robot_obs,
                    }

                    save_fn = f"{self.saving_frame_idx:06d}.npz"
                    save_path = os.path.join(self.saving_dir, save_fn)
                    self.data_manager.submit(frame_data_dict, self.saving_dir, file_path=None)
                    start_time = pygame.time.get_ticks()
                    # self.data_manager.save_data()
                    print("[DEBUG] Saving data cost:", pygame.time.get_ticks() - start_time)
                    print("---" * 30)
                    self.saving_frame_idx += 1

            # Framerate setting
            pygame.time.Clock().tick(300)
            pygame.display.flip()

        print("Program Exiting due to Exit Pressed")
        self.on_before_exit()
        self.game_state = GameState.STOPPED
        pygame.quit()
        self.robot.arm_power_off()
        self.on_after_exit()

    def debug_add_bar(self, label, value,
                      pos_x=50, pos_y=50,
                      min_value=-1, max_value=1,
                      bar_width=300, bar_height=30,
                      ):
        screen = self.screen
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        progress = (value - min_value) / (max_value - min_value)
        pygame.draw.rect(screen, RED, (pos_x, pos_y, bar_width, bar_height))
        pygame.draw.rect(screen, GREEN,
                         (pos_x, pos_y, progress * bar_width, bar_height))
        font = pygame.font.Font(None, 36)
        text = font.render(f"{label}: {value:.2f}", True, BLUE)
        screen.blit(text, (pos_x + bar_width + 20, pos_y + 5))

    ''' Read information from Joystick '''
    @abstractmethod
    def update_state(self) -> None: pass
    @abstractmethod
    def update_xyz(self) -> None: pass
    @abstractmethod
    def update_g(self) -> None: pass

    ''' Check which buttons are pressed on the Joystick '''
    @abstractmethod
    def is_exit_pressed(self) -> bool: pass
    @abstractmethod
    def is_back_pressed(self) -> bool: pass
    @abstractmethod
    def get_pause_button(self) -> int: pass
    @abstractmethod
    def get_angular_button(self) -> int: pass
    @abstractmethod
    def is_linear_jog_pressed(self) -> bool: pass
    @abstractmethod
    def is_angular_jog_pressed(self) -> bool: pass

    @abstractmethod
    def is_start_episode_pressed(self) -> bool: pass
    @abstractmethod
    def is_end_episode_pressed(self) -> bool: pass
    @abstractmethod
    def on_rumble(self) -> None: pass

    ''' Default behavior when some Events are detected '''
    def on_before_exit(self) -> None: pass
    def on_after_exit(self) -> None: pass
    def on_angular_button_released(self) -> None: pass

    def on_back_home(self) -> None:
        self.robot.joint_back_home()

    def switch_pause(self) -> None:
        print("Switching Game State")
        # Switch state between RUNNING and PAUSED
        if self.game_state == GameState.RUNNING:
            self.game_state = GameState.PAUSED
        elif self.game_state == GameState.PAUSED:
            self.game_state = GameState.RUNNING

    def on_linear_jog(self, xyz: dict = None) -> None:
        self.linear_xyz = xyz if xyz is not None else self.linear_xyz
        scaled_xyz = {k: v * self.linear_scale for k, v in self.linear_xyz.items()}
        self.robot.linear_jog_pub(scaled_xyz)

    def on_angular_jog(self, xyz: dict = None, z_39: float = None) -> None:
        self.angular_xyz = xyz if xyz is not None else self.angular_xyz
        self.z_39 = z_39 if z_39 is not None else self.z_39
        scaled_xyz = self.angular_xyz.copy()
        scaled_xyz['x'] = self.angular_xyz['y']
        scaled_xyz['y'] = -self.angular_xyz['x']
        scaled_xyz['z'] = scaled_xyz['z'] if abs(scaled_xyz['z']) > abs(self.z_39) else self.z_39
        scaled_xyz = {k: v * self.angular_scale for k, v in scaled_xyz.items()}
        self.robot.ang_jog_pub(scaled_xyz)

    def on_gripper_move(self, g: float = None) -> None:
        self.g = g if g is not None else self.g
        self.robot.gripper_set_pub(self.g)

    def on_start_episode(self) -> None:
        self.need_saving = True
        self.saving_dir = os.path.join(self.saving_root, self.get_datetime_str())
        self.saving_frame_idx = 0
        self.on_rumble()
        print("Episode Started")

    def on_end_episode(self) -> None:
        self.need_saving = False
        self.saving_dir = None
        self.saving_frame_idx = 0
        print("Episode Ended")
        self.on_rumble()
        print("Waiting for saving last data in queue")
        self.data_manager.close()
        print("Last data saved.")
        self.on_back_home()


class SwitchProController(BaseController):
    def __init__(self, robot: RobotClient, joystick_idx: int = 0,
                 saving_root: str = "collected_data_0422/",
                 **kwargs
                 ):
        BaseController.__init__(self, robot, saving_root=saving_root, **kwargs)
        """
        SwitchPro (Ubuntu)
        Left Stick 0639:    A1, A1, A0, A0
        Right Stick 0639:   A3, A3, A2, A2            
        A, B, X, Y:          1,  0,  2,  3
        LB/L, RB/R:          5,  6,             
        LT/ZL, RT/ZR:        7,  8,            
        screen:              4,
        menu/+:             10,
        share/-:             9,
        home:               11,
        """
        self.joystick_idx = joystick_idx
        self.joystick = pygame.joystick.Joystick(joystick_idx)
        self.joystick.init()

        self.linear_scale = 0.1
        self.angular_scale = 0.5
        print("[SwitchProController] Initialized")

    def get_joy_axis(self, axis: int, axis_eps: float = 8e-2):
        demand = self.joystick.get_axis(axis)
        if abs(demand) < axis_eps:
            demand = 0  # Applying dead band to avoid drift when joystick is released
        return demand

    def update_state(self):
        pass  # do nothing

    def update_xyz(self):
        """ stand as the robot, x:right, y:front, z:up
            face to the robot, x:left, y:front, z:up
        """
        x = -self.get_joy_axis(0)
        y = self.get_joy_axis(1)
        z = -self.get_joy_axis(3)
        z_39 = -self.get_joy_axis(2)
        self.linear_xyz = {'x': x, 'y': y, 'z': z}
        self.angular_xyz = {'x': x, 'y': y, 'z': z}
        self.z_39 = z_39

    def update_g(self):
        """ SwitchPro has no linear trigger (XBox S has) """
        if self.joystick.get_button(8):
            self.g = 1.  # pressed -> close
        else:
            self.g = 0.  # released -> open

    def is_exit_pressed(self):
        return self.joystick.get_button(11)

    def is_back_pressed(self):
        return self.joystick.get_button(0)

    def get_pause_button(self):
        return 10  # button number

    def get_angular_button(self):
        return 5

    def is_linear_jog_pressed(self):
        return self.joystick.get_button(6)

    def is_angular_jog_pressed(self):
        return self.joystick.get_button(5)

    def is_start_episode_pressed(self) -> bool:
        return self.joystick.get_button(2)

    def is_end_episode_pressed(self) -> bool:
        return self.joystick.get_button(3)

    def on_rumble(self) -> None:
        self.joystick.rumble(0.4, 0.4, 300)


class SwitchProIMUController(SwitchProController):
    def __init__(self, robot: RobotClient, joystick_idx: int = 0,
                 saving_root: str = "collected_data_0422/",
                 **kwargs
                 ):
        SwitchProController.__init__(self, robot, joystick_idx=joystick_idx,
                                     saving_root=saving_root, **kwargs)
        self.imu_controller = IMUControl()
        self.angular_scale = 1.5

    def update_xyz(self):
        x = -self.get_joy_axis(0)
        y = self.get_joy_axis(1)
        z = -self.get_joy_axis(3)
        self.linear_xyz = {'x': x, 'y': y, 'z': z}

        controller_quat, rpy_rel = self.imu_controller.capture_imu_pose()
        roll, pitch, yaw = rpy_rel
        self.angular_xyz = {'x': roll, 'y': pitch, 'z': yaw}
        self.z_39 = 0.

    def on_angular_jog(self, xyz: dict = None, z_39: float = None) -> None:
        """ Also let linear axes move """
        linear_scaled_xyz = {k: v * self.linear_scale for k, v in self.linear_xyz.items()}

        self.angular_xyz = xyz if xyz is not None else self.angular_xyz
        self.z_39 = z_39 if z_39 is not None else self.z_39
        scaled_xyz = self.angular_xyz.copy()
        scaled_xyz['x'] = self.angular_xyz['y']
        scaled_xyz['y'] = -self.angular_xyz['x']
        scaled_xyz['z'] = scaled_xyz['z'] if abs(scaled_xyz['z']) > abs(self.z_39) else self.z_39
        scaled_xyz = {k: v * self.angular_scale for k, v in scaled_xyz.items()}

        self.robot.lin_ang_jog_pub(linear_scaled_xyz, scaled_xyz)

    def on_angular_button_released(self):
        self.imu_controller.rpy_rel = [0., 0., 0.]


class PS5DualSenseController(BaseController):
    def __init__(self, robot: RobotClient, joystick_idx: int = 0,
                 saving_root: str = "collected_data_0422/",
                 **kwargs
                 ):
        BaseController.__init__(self, robot, saving_root=saving_root, **kwargs)
        """
        PS5DualSense (Ubuntu)
        Left Stick 0639:    A1, A1, A0, A0
        Right Stick 0639:   A4, A5, A3, A3
        A, B, X, Y:          1,  0,  2,  3
        LB/L1, RB/R1:        4,  5,
        LT/L2, RT/R2:        6,  7,
        screen:              8,
        menu:                9,
        share:              10,
        """
        self.joystick_idx = joystick_idx
        self.joystick = pygame.joystick.Joystick(joystick_idx)
        self.joystick.init()

        self.linear_scale = 0.1
        self.angular_scale = 0.5
        print("[PS5DualSenseController] Initialized")

    def get_joy_axis(self, axis: int, axis_eps: float = 8e-2):
        demand = self.joystick.get_axis(axis)
        if abs(demand) < axis_eps:
            demand = 0  # Applying dead band to avoid drift when joystick is released
        return demand

    def update_state(self):
        pass  # do nothing

    def update_xyz(self):
        """ stand as the robot, x:right, y:front, z:up
            face to the robot, x:left, y:front, z:up
        """
        x = -self.get_joy_axis(0)
        y = self.get_joy_axis(1)
        z = -self.get_joy_axis(4)
        z_39 = -self.get_joy_axis(3)
        self.linear_xyz = {'x': x, 'y': y, 'z': z}
        self.angular_xyz = {'x': x, 'y': y, 'z': z}
        self.z_39 = z_39

    def update_g(self):
        """ Although DualSense has linear trigger, but there is no need to use it """
        if self.joystick.get_button(7):
            self.g = 1.  # pressed -> close
        else:
            self.g = 0.  # released -> open

    def is_exit_pressed(self):
        return self.joystick.get_button(10)

    def is_back_pressed(self):
        return self.joystick.get_button(0)

    def get_pause_button(self):
        return 9  # button number

    def get_angular_button(self):
        return 4

    def is_linear_jog_pressed(self):
        return self.joystick.get_button(5)

    def is_angular_jog_pressed(self):
        return self.joystick.get_button(4)

    def is_start_episode_pressed(self) -> bool:
        return self.joystick.get_button(2)

    def is_end_episode_pressed(self) -> bool:
        return self.joystick.get_button(3)

    def on_rumble(self) -> None:
        self.joystick.rumble(0.2, 0.4, 100)


class PS5DualSenseIMUController(PS5DualSenseController):
    def __init__(self, robot: RobotClient, joystick_idx: int = 0,
                 saving_root: str = "collected_data_0422/",
                 **kwargs
                 ):
        PS5DualSenseController.__init__(self, robot, joystick_idx=joystick_idx,
                                     saving_root=saving_root, **kwargs)
        self.imu_controller = RawIMUHandler()
        self.angular_scale = .3

    def update_xyz(self):
        x = -self.get_joy_axis(0)
        y = self.get_joy_axis(1)
        z = -self.get_joy_axis(4)
        self.linear_xyz = {'x': x, 'y': y, 'z': z}

        # start = time.time()
        # euler_data = self.imu_controller.get_latest_euler()
        # print(f"[PS5DualSenseIMUController] Euler data received. time={(time.time()-start) * 1000:.2f} ms")
        # # rpy_now = euler_data['euler'][:3]
        # rpy_rel = euler_data['euler'][3:6]
        # roll, pitch, yaw = rpy_rel
        # self.angular_xyz = {'x': roll, 'y': pitch, 'z': yaw}
        # self.z_39 = 0.

    def on_angular_jog(self, xyz: dict = None, z_39: float = None) -> None:
        """ Also let linear axes move """
        linear_scaled_xyz = {k: v * self.linear_scale for k, v in self.linear_xyz.items()}

        # Only need to obtain angular_xyz when moving
        euler_data = self.imu_controller.get_latest_euler()
        # rpy_now = euler_data['euler'][:3]
        rpy_rel = euler_data['euler'][3:6]
        roll, pitch, yaw = rpy_rel
        self.angular_xyz = {'x': -roll, 'y': -pitch, 'z': yaw}
        self.z_39 = 0.

        self.angular_xyz = xyz if xyz is not None else self.angular_xyz
        self.z_39 = z_39 if z_39 is not None else self.z_39
        scaled_xyz = self.angular_xyz.copy()
        scaled_xyz['x'] = self.angular_xyz['y']
        scaled_xyz['y'] = -self.angular_xyz['x']
        scaled_xyz['z'] = scaled_xyz['z'] if abs(scaled_xyz['z']) > abs(self.z_39) else self.z_39
        scaled_xyz = {k: v * self.angular_scale for k, v in scaled_xyz.items()}

        self.robot.lin_ang_jog_pub(linear_scaled_xyz, scaled_xyz)

    def on_angular_button_released(self):
        self.imu_controller.reset_pose()

    def on_before_exit(self) -> None:
        self.imu_controller.stop()


class KeyboardController:
    def __init__(self):
        mapping = {
            'left_stick_0639': [
                pygame.K_w,
                pygame.K_s,
                pygame.K_a,
                pygame.K_d,
            ],
            'right_stick_0639': [
                pygame.K_UP,
                pygame.K_DOWN,
                pygame.K_LEFT,
                pygame.K_RIGHT,
            ],
            'abxy': [
                pygame.K_j,
                pygame.K_k,
                pygame.K_i,
                pygame.K_o,
            ],
            'screen': pygame.K_b,
            'menu': pygame.K_m,
            'share': pygame.K_n,
            'LB': pygame.K_LSHIFT,
            'RB': pygame.K_SPACE,
            'RT': pygame.K_RETURN
        }
        self.mapping = mapping

        # Dynamic
        self.keys = pygame.key.get_pressed()
        self.xyz = {'x': 0.0,
                  'y': 0.0,
                  'z': 0.0}

    def get_state(self):
        self.keys = pygame.key.get_pressed()
        return self.keys

    def get_xyz(self, keys):
        coords = self.xyz

        if keys[self.mapping['left_stick_0639'][0]]:
            coords['x'] = -1
        elif keys[self.mapping['left_stick_0639'][1]]:
            coords['x'] = 1
        elif keys[self.mapping['left_stick_0639'][2]]:
            coords['y'] = -1
        elif keys[self.mapping['left_stick_0639'][3]]:
            coords['y'] = 1
        elif keys[self.mapping['right_stick_0639'][0]]:
            coords['z'] = 1
        elif keys[self.mapping['right_stick_0639'][1]]:
            coords['z'] = -1

        self.xyz = coords
        return coords

    def check_exit(self):
        keys = self.keys
        if keys[self.mapping['abxy'][1]]:
            return True
        return False

    def check_linear_move(self):
        keys = self.keys
        # No need to check RB here

    def check_jog_move(self):
        keys = self.keys
        

