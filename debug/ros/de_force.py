#!/usr/bin/env python3
"""
Robotiq FT 300 Force/Torque Sensor Reader
Based on the official C implementation
"""

import time
import sys
import serial
import struct

from robokit.data_manager.ft300_handler import FT300Handler as FT300

# Draw a dynamic visual separator line
# Which task is closer to force??

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port_prefix', default='/dev/ttyUSB', help='Serial port prefix    ')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    parser.add_argument('--raw', action='store_true', help='Show raw packets')
    args = parser.parse_args()

    sensor = FT300(
        port_prefix=args.port_prefix,
        port_id=0,
    )

    for attempt in range(5):
        if sensor.connect():
            break
        print(f"Retrying connection... ({attempt + 1}/5)")
        time.sleep(1)
    else:
        print("Failed to connect after 5 attempts")
        sys.exit(1)

    print("\nReading sensor data_manager... Press Ctrl+C to stop\n")
    print("Expected format: ( Fx[N] , Fy[N] , Fz[N] , Mx[Nm] , My[Nm] , Mz[Nm] )")
    print("-" * 70)

    try:
        read_count = 0
        error_count = 0

        while True:
            data = sensor.read_ft()

            if data:
                # Display in same format as driverSensor
                print(f"( {data[0]:8.6f} , {data[1]:8.6f} , {data[2]:8.6f} , "
                      f"{data[3]:7.6f} , {data[4]:7.6f} , {data[5]:7.6f} )")
                read_count += 1
                error_count = 0

                if args.debug and read_count % 100 == 0:
                    print(f"[Debug: {read_count} packets read successfully]")
            else:
                error_count += 1
                if args.debug and error_count % 100 == 0:
                    print(f"[Debug: No valid packet... attempt {error_count}]")

            time.sleep(0.01)  # ~100Hz reading rate

    except KeyboardInterrupt:
        print(f"\n\nStopping... (Successfully read {read_count} packets)")
    finally:
        sensor.close()


if __name__ == '__main__':
    main()