#!/usr/bin/env python3
"""
Robotiq FT 300 Force/Torque Sensor Reader
Based on the official C implementation
"""

import time
import sys
import serial
import struct
import numpy as np


class FT300Handler:
    def __init__(self, port_prefix: str = '/dev/ttyUSB', port_id: int = 0):
        self.port_prefix = port_prefix
        self.port_id = port_id
        self.port = f"{self.port_prefix}{self.port_id}"
        self.ser = None
        self.buffer = b''
        self.offset = np.zeros(6)  # Added for drift/offset compensation

    def connect(self, calibrate_samples: int = 200):
        for _ in range(5):
            if self._try_connect():
                self.calibrate(num_samples=calibrate_samples)  # Calibrate automatically upon successful connection
                return True

    def _try_connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=19200,
                bytesize=8,
                stopbits=1,
                parity='N',
                timeout=0.1
            )
            time.sleep(0.2)
            print(f"Connected to {self.port}")
            return True
        except Exception as e:
            # print(f"Connection error: {e}")
            self.port_id += 1
            self.port = f"{self.port_prefix}{self.port_id}"
            return False

    def calibrate(self, num_samples=50):
        """
        Read initial samples to determine the zero-offset (drift).
        This should be called when the robot is stationary.
        """
        print("Calibrating FT300 sensor... Please do not move the robot.")
        samples = []
        while len(samples) < num_samples:
            data = self.find_and_parse_packet(apply_offset=False)  # Read raw data
            if data:
                samples.append(data)
            time.sleep(0.02)  # ~50Hz

        if samples:
            self.offset = np.mean(samples, axis=0)
            print(f"Calibration complete. Offset set to: {self.offset}")
        else:
            print("Warning: Failed to collect samples for FT300 calibration. Offset remains zero.")

    def compute_crc(self, data):
        """Compute CRC-16 for Modbus RTU (same as in C code)"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def find_and_parse_packet(self, apply_offset=True):
        """Find and parse a valid packet from the stream"""
        # Read available data
        new_data = self.ser.read(self.ser.in_waiting or 100)
        self.buffer += new_data

        # Need at least 16 bytes
        while len(self.buffer) >= 16:
            # Look for header 0x20 0x4E in the buffer
            header_pos = -1

            # Search for 0x20 0x4E pattern
            for i in range(len(self.buffer) - 15):
                if self.buffer[i] == 0x20 and self.buffer[i + 1] == 0x4E:
                    header_pos = i
                    break

            if header_pos == -1:
                # No header found, keep last 15 bytes
                self.buffer = self.buffer[-15:]
                return None

            # Check if we have enough data for a complete message (14 bytes after header)
            if header_pos + 16 > len(self.buffer):
                # Need more data
                return None

            # Extract the message (header + 12 data bytes + 2 CRC bytes)
            message = self.buffer[header_pos:header_pos + 16]

            # Verify CRC
            data_for_crc = message[:14]  # Everything except CRC
            computed_crc = self.compute_crc(data_for_crc)
            received_crc = message[14] | (message[15] << 8)

            if computed_crc == received_crc:
                # Valid packet! Remove it from buffer
                self.buffer = self.buffer[header_pos + 16:]

                # Parse the data (6 signed 16-bit values in little-endian)
                values = []
                for j in range(6):
                    byte_low = message[2 + 2 * j]
                    byte_high = message[3 + 2 * j]
                    # Reconstruct signed 16-bit value
                    raw_value = byte_low + (byte_high << 8)
                    # Convert to signed
                    if raw_value > 32767:
                        raw_value -= 65536

                    # Scale: Forces (0-2) /100, Torques (3-5) /1000
                    if j < 3:
                        values.append(raw_value / 100.0)
                    else:
                        values.append(raw_value / 1000.0)

                values = np.array(values)
                if apply_offset:
                    values -= self.offset

                return tuple(values)
            else:
                # Bad CRC, skip this byte and continue searching
                self.buffer = self.buffer[header_pos + 1:]

        return None

    def read_ft(self, flush_old=True):
        """
        读取力/力矩数据
        Args:
            flush_old: 是否清空旧数据，只读最新的
        """
        if flush_old:
            # 清空串口和内部缓冲区中的旧数据
            self._flush_old_data()

        return self.find_and_parse_packet()

    def _flush_old_data(self):
        """清空旧数据，确保读取最新"""
        # 方法 1：重置串口输入缓冲区
        if self.ser:
            self.ser.reset_input_buffer()

        # 方法 2：清空内部缓冲区
        self.buffer = b''

        # 方法 3：读取并丢弃所有待处理数据
        # 然后读一个新的完整包
        waiting = self.ser.in_waiting
        if waiting > 0:
            _ = self.ser.read(waiting)

    def close(self):
        if self.ser:
            self.ser.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    parser.add_argument('--raw', action='store_true', help='Show raw packets')
    args = parser.parse_args()

    sensor = FT300Handler(args.port)

    if not sensor.connect():
        print("Failed to connect")
        sys.exit(1)

    print("\nReading sensor data... Press Ctrl+C to stop\n")
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
