import time
import enum
from abc import ABC, abstractmethod
import pygame

from .robot_client import RobotClient


class GameState(enum.Enum):
    INIT = 0
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


class BaseController:
    def __init__(self, robot: RobotClient):
        pygame.init()
        pygame.joystick.init()

        self.robot = robot
        self.linear_scale = 0.1
        self.angular_scale = 0.1

        # Dynamic
        self.xyz = {'x': 0.0, 'y': 0.0, 'z': 0.0}
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

    def start(self):
        self.game_state = GameState.RUNNING
        while self.game_state != GameState.STOPPED:
            for event in pygame.event.get():
                time.sleep(0)

            self.update_state()
            self.update_xyz()
            self.update_g()

            # print('[DEBUG]', self.robot.Pose, self.robot.State)

            # Check Paused
            if self.is_pause_pressed():
                self.on_pause()
            if self.game_state == GameState.PAUSED:
                continue

            # Check Exit
            if self.is_exit_pressed():
                self.game_state = GameState.STOPPED

            # Check Back Home
            if self.is_back_pressed():
                self.on_back_home()

            # Check XYZ Move
            if self.is_linear_jog_pressed():
                self.on_linear_jog()
            elif self.is_angular_jog_pressed():
                self.on_angular_jog()
            else:
                self.xyz = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                self.on_linear_jog()
                self.on_angular_jog()

            # Check Gripper
            self.on_gripper_move()

            # Framerate setting
            pygame.time.Clock().tick(60)

        print("Program Exiting due to Exit Pressed")
        self.on_before_exit()
        self.game_state = GameState.STOPPED
        pygame.quit()
        self.robot.arm_power_off()
        self.on_after_exit()

    @abstractmethod
    def update_state(self) -> None: pass
    @abstractmethod
    def update_xyz(self) -> None: pass
    @abstractmethod
    def update_g(self) -> None: pass

    @abstractmethod
    def is_exit_pressed(self) -> bool: pass
    @abstractmethod
    def is_back_pressed(self) -> bool: pass
    @abstractmethod
    def is_pause_pressed(self) -> bool: pass
    @abstractmethod
    def is_linear_jog_pressed(self) -> bool: pass
    @abstractmethod
    def is_angular_jog_pressed(self) -> bool: pass

    def on_before_exit(self) -> None: pass
    def on_after_exit(self) -> None: pass

    def on_back_home(self) -> None:
        self.robot.joint_back_home()

    def on_pause(self) -> None:
        # Switch state between RUNNING and PAUSED
        if self.game_state == GameState.RUNNING:
            self.game_state = GameState.PAUSED
        elif self.game_state == GameState.PAUSED:
            self.game_state = GameState.RUNNING

    def on_linear_jog(self, xyz: dict = None) -> None:
        self.xyz = xyz if xyz is not None else self.xyz
        scaled_xyz = {k: v * self.linear_scale for k, v in self.xyz.items()}
        self.robot.linear_jog_pub(scaled_xyz)

    def on_angular_jog(self, xyz: dict = None) -> None:
        self.xyz = xyz if xyz is not None else self.xyz
        scaled_xyz = self.xyz.copy()
        scaled_xyz['x'] = self.xyz['y']
        scaled_xyz['y'] = -self.xyz['x']
        scaled_xyz['z'] = scaled_xyz['z'] if abs(scaled_xyz['z']) > abs(self.z_39) else self.z_39
        scaled_xyz = {k: v * self.angular_scale for k, v in scaled_xyz.items()}
        self.robot.ang_jog_pub(scaled_xyz)

    def on_gripper_move(self, g: float = None) -> None:
        self.g = g if g is not None else self.g
        self.robot.gripper_set_pub(self.g)


class SwitchProController(BaseController):
    def __init__(self, robot: RobotClient, joystick_idx: int = 0):
        BaseController.__init__(self, robot)
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
        self.xyz = {'x': x, 'y': y, 'z': z}
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

    def is_pause_pressed(self):
        return self.joystick.get_button(10)

    def is_linear_jog_pressed(self):
        return self.joystick.get_button(6)

    def is_angular_jog_pressed(self):
        return self.joystick.get_button(5)


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
        

