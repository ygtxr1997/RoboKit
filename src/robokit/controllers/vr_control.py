import time
import threading
import time
import os
import sys

import numpy as np
from ppadb.client import Client as AdbClient
import transforms3d
from transforms3d.quaternions import qmult, qinverse, mat2quat
from transforms3d.euler import quat2euler


class FPSCounter:
    def __init__(self):
        current_time = time.time()
        self.start_time_for_display = current_time
        self.last_time = current_time
        self.x = 5  # displays the frame rate every X second
        self.time_between_calls = []
        self.elements_for_mean = 50

    def getAndPrintFPS(self, print_fps=True):
        current_time = time.time()
        self.time_between_calls.append(1.0/(current_time - self.last_time + 1e-9))
        if len(self.time_between_calls) > self.elements_for_mean:
            self.time_between_calls.pop(0)
        self.last_time = current_time
        frequency = np.mean(self.time_between_calls)
        if (current_time - self.start_time_for_display) > self.x and print_fps:
            print("Frequency: {}Hz".format(int(frequency)))
            self.start_time_for_display = current_time
        return frequency


def parse_buttons(text):
    split_text = text.split(',')
    buttons = {}
    if 'R' in split_text:  # right hand if available
        split_text.remove('R')  # remove marker
        buttons.update({'A': False,
                        'B': False,
                        'RThU': False,  # indicates that right thumb is up from the rest position
                        'RJ': False,  # joystick pressed
                        'RG': False,  # boolean value for trigger on the grip (delivered by SDK)
                        'RTr': False  # boolean value for trigger on the index finger (delivered by SDK)
                        })
        # besides following keys are provided:
        # 'rightJS' / 'leftJS' - (x, y) position of joystick. x, y both in range (-1.0, 1.0)
        # 'rightGrip' / 'leftGrip' - float value for trigger on the grip in range (0.0, 1.0)
        # 'rightTrig' / 'leftTrig' - float value for trigger on the index finger in range (0.0, 1.0)

    if 'L' in split_text:  # left hand accordingly
        split_text.remove('L')  # remove marker
        buttons.update({'X': False, 'Y': False, 'LThU': False, 'LJ': False, 'LG': False, 'LTr': False})
    for key in buttons.keys():
        if key in list(split_text):
            buttons[key] = True
            split_text.remove(key)
    for elem in split_text:
        split_elem = elem.split(' ')
        if len(split_elem) < 2:
            continue
        key = split_elem[0]
        value = tuple([float(x) for x in split_elem[1:]])
        buttons[key] = value
    return buttons


def eprint(*args, **kwargs):
    RED = "\033[1;31m"
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)


class OculusReader:
    def __init__(self,
            ip_address=None,
            port = 5555,
            APK_name='com.rail.oculus.teleop',
            print_FPS=False,
            run=True
        ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        self._lock = threading.Lock()
        self.tag = 'wE9ryARX'

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        if self.print_FPS:
            self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self):
        self.running = True
        self.device.shell('am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER')
        self.thread = threading.Thread(target=self.device.shell, args=("logcat -T 0", self.read_logcat_by_line))
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system('adb devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry==1:
                os.system('adb tcpip ' + str(self.port))
            if retry==2:
                eprint('Make sure that device is running and is available at the IP address specified as the OculusReader argument `ip_address`.')
                eprint('Currently provided IP address:', self.ip_address)
                eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_device(client=client, retry=retry+1)
        return device

    def get_usb_device(self, client):
        try:
            devices = client.devices()
        except RuntimeError:
            os.system('adb devices')
            devices = client.devices()
        for device in devices:
            if device.serial.count('.') < 3:
                return device
        eprint('Device not found. Make sure that device is running and is connected over USB')
        eprint('Run `adb devices` to verify that the device is visible.')
        exit(1)

    def get_device(self):
        # Default is "127.0.0.1" and 5037
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    def install(self, APK_path=None, verbose=True, reinstall=False):
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'APK', 'teleop-debug.apk')
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print('APK installed successfully.')
                else:
                    eprint('APK install failed.')
            elif verbose:
                print('APK is already installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print('APK uninstall finished.')
                    print('Please verify if the app disappeared from the list as described in "UNINSTALL.md".')
                    print('For the resolution of this issue, please follow https://github.com/Swind/pure-python-adb/issues/71.')
                else:
                    eprint('APK uninstall failed')
            elif verbose:
                print('APK is not installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    @staticmethod
    def process_data(string):
        try:
            transforms_string, buttons_string = string.split('&')
        except ValueError:
            return None, None
        split_transform_strings = transforms_string.split('|')
        transforms = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4,4))
            pair = pair_string.split(':')
            if len(pair) != 2:
                continue
            left_right_char = pair[0] # is r or l
            transform_string = pair[1]
            values = transform_string.split(' ')
            c = 0
            r = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[r][c] = float(value)
                c += 1
                if c >= 4:
                    c = 0
                    r += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def extract_data(self, line):
        output = ''
        if self.tag in line:
            try:
                output += line.split(self.tag + ': ')[1]
            except ValueError:
                pass
        return output

    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()


class QuestHandler(OculusReader):
    def __init__(self,
                 ip_address=None,
                 port=5555,
                 APK_name='com.rail.oculus.teleop',
                 print_FPS=False,
                 run=True
                 ):
        self.xyz_last = np.zeros(3)  # last XYZ position
        self.xyz_now = np.zeros(3)  # current XYZ position
        self.xyz_rel = np.zeros(3)  # xyz_now <- xyz_last + xyz_rel

        self.q_last = np.array([1.0, 0.0, 0.0, 0.0])  # last pose, output by get_latest_euler()
        self.q_now = np.array([1.0, 0.0, 0.0, 0.0])  # current pose
        self.q_rel = np.array([1.0, 0.0, 0.0, 0.0])  # q_now <- q_last + q_rel
        self.rpy_rel = np.zeros(3)  # q_now <- q_last + rpy_rel
        self.rpy_now = np.zeros(3)  # 'sxyz' order of current pose ``q_now``

        super(QuestHandler, self).__init__(ip_address, port, APK_name, print_FPS, run)

        self._reset_evt = False
        self._acquire_euler_evt = False

    def _init_state(self):
        self.xyz_last = np.zeros(3)  # last XYZ position
        self.xyz_now = np.zeros(3)  # current XYZ position
        self.xyz_rel = np.zeros(3)  # xyz_now <- xyz_last + xyz_rel

        self.q_last = np.array([1.0, 0.0, 0.0, 0.0])  # last pose, output by get_latest_euler()
        self.q_now = np.array([1.0, 0.0, 0.0, 0.0])  # current pose
        self.q_rel = np.array([1.0, 0.0, 0.0, 0.0])  # q_now <- q_last + q_rel
        self.rpy_rel = np.zeros(3)  # q_now <- q_last + rpy_rel
        self.rpy_now = np.zeros(3)  # 'sxyz' order of current pose ``q_now``

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            # Listening to events
            if self._reset_evt:
                with self._lock:
                    self._init_state()
                    self._reset_evt = False

            if self._acquire_euler_evt:
                # Calculate: rpy_rel, xyz_rel
                # Set: q_last, xyz_last
                with self._lock:
                    self._acquire_euler_evt = False
                    self.q_rel = d_q_rot = qmult(self.q_now, qinverse(self.q_last))
                    self.rpy_rel = np.degrees(quat2euler(d_q_rot, axes='sxyz'))
                    self.q_last = self.q_now

                    d_xyz = self.xyz_now - self.xyz_last
                    self.xyz_rel = d_xyz
                    self.xyz_last = self.xyz_now
            ##########

            try:
                line = file_obj.readline().strip()
                print(line)
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)

                    # Accumulate delta_xyz and delta_euler
                    left_transform, right_transform = transforms['l'], transforms['r']
                    d_q_rotation = mat2quat(right_transform[:3, :3])  # (3x3) -> quat
                    d_translation = right_transform[:3, 3]  # (3x1) -> (3,)

                    with self._lock:
                        # self.q_rel = qmult(d_q_rotation, qinverse(self.q_now))
                        self.q_now = d_q_rotation  # qmult(self.q_now, d_q_rotation)
                        self.rpy_now = np.degrees(quat2euler(self.q_now, axes='sxyz'))
                        self.xyz_now = d_translation  # self.xyz_now + d_translation

                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()

    def reset_pose(self):
        with self._lock:
            self._reset_evt = True

    def get_latest_euler(self):
        self._acquire_euler_evt = True
        with self._lock:
            return {
                'euler': self.rpy_rel,
                'quat': self.q_rel,
                'xyz': self.xyz_rel,
                'now_euler': self.rpy_now,
                'now_xyz': self.xyz_now,
                'now_quat': self.q_now,
            }

    def get_last_buttons(self):
        with self._lock:
            return self.last_buttons

    def is_right_gripper_pressed(self):
        ''' {'A': False, 'B': False, 'RThU': False, 'RJ': False, 'RG': False, 'RTr': False,
        'X': False, 'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False,
        'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,),
        'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (0.0,)}
        '''
        last_buttons = self.get_last_buttons()
        return last_buttons['RG']


def main():
    # # Op1. OculusReader
    # oculus_reader = OculusReader()
    #
    # while True:
    #     time.sleep(0.3)
    #     print(oculus_reader.get_transformations_and_buttons())

    # Op2. QuestHandler
    quest_handler = QuestHandler()
    while True:
        time.sleep(0.1)
        print(quest_handler.get_latest_euler())


if __name__ == '__main__':
    main()
