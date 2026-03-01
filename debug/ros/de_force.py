#!/usr/bin/env python3
"""
Robotiq FT 300 Force/Torque Sensor Reader
Based on the official C implementation
"""
import copy
import time
import sys
import serial
import struct

from robokit.data_manager.ft300_handler import FT300ModbusStream as FT300
from robokit.debug_utils.images import DynamicDataDrawer
from robokit.debug_utils.times import time_stat

# Draw a dynamic visual separator line
# Which task is closer to force??


class ForceTorqueVisualizer(DynamicDataDrawer):
    def __init__(self, max_points: int = 200):
        self.ft_sensor = FT300()
        super().__init__(data_provider=self.ft_sensor,
                         data_keys=[
                             ['Fx', 'Fy', 'Fz'],
                             ['Mx', 'My', 'Mz'],
                         ],
                         y_minmax_values=[
                             (-50, 50), (-50, 50), (-50, 50),
                             (-5, 5), (-5, 5), (-5, 5),
                         ],
                         max_points=max_points)
        self.ft_sensor.connect(calibrate_samples=300)

    def get_new_data(self) -> dict:
        force_data = self.ft_sensor.read_ft()
        force_xyz = force_data[:3]
        torque_xyz = force_data[3:6]

        vis_data = {
            'Fx': float(force_xyz[0]),
            'Fy': float(force_xyz[1]),
            'Fz': float(force_xyz[2]),
            'Mx': float(torque_xyz[0]),
            'My': float(torque_xyz[1]),
            'Mz': float(torque_xyz[2]),
        }
        return vis_data



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port_prefix', default='/dev/ttyUSB', help='Serial port prefix    ')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    parser.add_argument('--raw', action='store_true', help='Show raw packets')
    args = parser.parse_args()

    ft_visualizer = ForceTorqueVisualizer()
    try:
        ft_visualizer.run()
    except KeyboardInterrupt:
        print("Exiting by KeyboardInterrupt")
    finally:
        ft_visualizer.ft_sensor.close()
        print("FT sensor closed")

    # sensor = FT300(
    #     port_prefix=args.port_prefix,
    #     port_id=0,
    # )
    #
    # for attempt in range(5):
    #     if sensor.connect():
    #         break
    #     print(f"Retrying connection... ({attempt + 1}/5)")
    #     time.sleep(1)
    # else:
    #     print("Failed to connect after 5 attempts")
    #     sys.exit(1)
    #
    # print("\nReading sensor data_manager... Press Ctrl+C to stop\n")
    # print("Expected format: ( Fx[N] , Fy[N] , Fz[N] , Mx[Nm] , My[Nm] , Mz[Nm] )")
    # print("-" * 70)
    #
    # try:
    #     read_count = 0
    #     error_count = 0
    #
    #     while True:
    #         with time_stat("FT300 read_ft"):
    #             data = sensor.read_ft()
    #
    #         if data:
    #             # Display in same format as driverSensor
    #             # print(f"( {data[0]:8.6f} , {data[1]:8.6f} , {data[2]:8.6f} , "
    #             #       f"{data[3]:7.6f} , {data[4]:7.6f} , {data[5]:7.6f} )")
    #             if read_count % 10 == 0:
    #                 time_stat.print_stats()
    #
    #             read_count += 1
    #             error_count = 0
    #
    #             if args.debug and read_count % 100 == 0:
    #                 print(f"[Debug: {read_count} packets read successfully]")
    #         else:
    #             error_count += 1
    #             if args.debug and error_count % 100 == 0:
    #                 print(f"[Debug: No valid packet... attempt {error_count}]")
    #
    #         # time.sleep(0.01)  # ~100Hz reading rate
    #
    # except KeyboardInterrupt:
    #     print(f"\n\nStopping... (Successfully read {read_count} packets)")
    # finally:
    #     sensor.close()


if __name__ == '__main__':
    main()