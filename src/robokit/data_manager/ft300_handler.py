#!/usr/bin/env python3
"""
Robotiq FT 300 Force/Torque Sensor Reader
Based on the official C implementation
"""

import time
import sys
import serial
import struct


class FT300Handler:
    def __init__(self, port_prefix: str = '/dev/ttyUSB', port_id: int = 0):
        self.port_prefix = port_prefix
        self.port_id = port_id
        self.port = f"{self.port_prefix}{self.port_id}"
        self.ser = None
        self.buffer = b''

    def connect(self):
        for _ in range(5):
            if self._try_connect():
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

    def find_and_parse_packet(self):
        """Find and parse a valid packet from the stream"""
        # Read available data_manager
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

            # Check if we have enough data_manager for a complete message (14 bytes after header)
            if header_pos + 16 > len(self.buffer):
                # Need more data_manager
                return None

            # Extract the message (header + 12 data_manager bytes + 2 CRC bytes)
            message = self.buffer[header_pos:header_pos + 16]

            # Verify CRC
            data_for_crc = message[:14]  # Everything except CRC
            computed_crc = self.compute_crc(data_for_crc)
            received_crc = message[14] | (message[15] << 8)

            if computed_crc == received_crc:
                # Valid packet! Remove it from buffer
                self.buffer = self.buffer[header_pos + 16:]

                # Parse the data_manager (6 signed 16-bit values in little-endian)
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

                return tuple(values)
            else:
                # Bad CRC, skip this byte and continue searching
                self.buffer = self.buffer[header_pos + 1:]

        return None

    def read_ft(self):
        """Read force/torque data_manager"""
        return self.find_and_parse_packet()

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