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

from robokit.debug_utils.times import time_stat


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
        """Find and parse a valid packet from the stream, returning the latest one found."""
        bytes_to_read = self.ser.in_waiting
        if bytes_to_read > 0:
            new_data = self.ser.read(bytes_to_read)
            self.buffer += new_data

        latest_packet = None
        # Loop to parse all complete packets in the buffer
        while True:
            if len(self.buffer) < 16:
                break  # Not enough data for a packet, exit loop

            header_pos = self.buffer.find(b'\x20\x4E')

            if header_pos == -1:
                # No header found, keep last 15 bytes in case header is split
                self.buffer = self.buffer[-15:]
                break

            # If a header is found, but not at the start, discard the garbage before it
            if header_pos > 0:
                self.buffer = self.buffer[header_pos:]

            # Check if we have a full packet from the header position
            if len(self.buffer) < 16:
                break  # Not enough data after the header, wait for more

            message = self.buffer[:16]
            data_for_crc = message[:14]
            computed_crc = self.compute_crc(data_for_crc)
            received_crc = message[14] | (message[15] << 8)

            if computed_crc == received_crc:
                # Valid packet found, parse it
                values = []
                for j in range(6):
                    raw_value = struct.unpack_from('<h', message, 2 + 2 * j)[0]
                    scale = 100.0 if j < 3 else 1000.0
                    values.append(raw_value / scale)

                values = np.array(values)
                if apply_offset:
                    values -= self.offset

                latest_packet = tuple(values)  # Store the valid packet

                # Consume the packet from the buffer and continue to look for more
                self.buffer = self.buffer[16:]
            else:
                # Bad CRC, discard the header byte and search again
                self.buffer = self.buffer[1:]

        return latest_packet

    def read_ft_old(self, flush_old=True):
        """
        读取力/力矩数据
        Args:
            flush_old: 是否清空旧数据，只读最新的
        """
        if flush_old:
            with time_stat("FT300 flush_old_data", enabled=False):
                # 清空串口和内部缓冲区中的旧数据
                self._flush_old_data()

        with time_stat("FT300 read_data", enabled=False):
            result = self.find_and_parse_packet()

        return result

    def read_ft(self):
        """
        Reads the latest force/torque data with minimal latency.
        This method blocks until it can return a valid packet.
        """
        with time_stat("FT300 read_data", enabled=False):
            while True:
                # 1. Attempt to parse the latest packet from the current buffer.
                #    This call will consume all available full packets.
                packet = self.find_and_parse_packet()
                if packet:
                    # Since find_and_parse_packet returns the *last* packet found,
                    # we can return it immediately.
                    # time_stat.print_stats()
                    return packet

                # 2. If no packet was found, block and wait for at least one new byte.
                #    This is more efficient than sleeping.
                new_data = self.ser.read(1)
                if new_data:
                    self.buffer += new_data
                # Loop will continue, and the new data will be processed in the next iteration.

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


class FT300ModbusStream(FT300Handler):
    """
    An enhanced handler for the FT300 sensor that explicitly controls the data stream
    using Modbus RTU commands, as recommended by the official documentation for
    robust operation.

    It sends a command to start the stream on connection and a command to stop
    it on close, providing more reliable communication than relying on the
    sensor's default behavior.
    """

    def __init__(self, port_prefix: str = '/dev/ttyUSB', port_id: int = 0, slave_id: int = 0x09):
        """
        Initializes the handler.

        Args:
            port_prefix: The prefix for the serial port (e.g., '/dev/ttyUSB').
            port_id: The starting ID number for the serial port.
            slave_id: The Modbus slave ID of the sensor, default is 9.
        """
        super().__init__(port_prefix, port_id)
        self.slave_id = slave_id

    def connect(self, calibrate_samples: int = 200):
        """
        Connects to the sensor, explicitly starts the data stream via a Modbus
        command, and then performs calibration.
        """
        for _ in range(5):
            if self._try_connect():
                print("Sending Modbus command to start data stream...")
                # Command to start stream: Write 0x0200 to register 410 (0x019A)
                # Format: [SlaveID, FC, Addr_Hi, Addr_Lo, Num_Reg_Hi, Num_Reg_Lo, Byte_Count, Data_Hi, Data_Lo, CRC_Lo, CRC_Hi]
                # The CRC (CD CA) is pre-calculated for the default command with SlaveID 9.
                # If you change the SlaveID, the CRC must be recalculated.
                start_stream_cmd = bytes([
                    self.slave_id, 0x10, 0x01, 0x9A, 0x00, 0x01, 0x02, 0x02, 0x00, 0xCD, 0xCA
                ])

                self.ser.write(start_stream_cmd)
                time.sleep(0.1)  # Give sensor time to process the command
                self._flush_old_data()  # Clear any Modbus response from the buffer
                print("Data stream started.")

                self.calibrate(num_samples=calibrate_samples)
                return True

        print("Failed to connect to FT300 sensor.")
        return False

    def close(self):
        """
        Explicitly stops the data stream by sending a sequence of 0xFF bytes
        before closing the serial port.
        """
        if self.ser and self.ser.is_open:
            print("Sending command to stop data stream...")
            # As per documentation, send ~50 0xFF bytes to stop the stream
            stop_cmd = b'\xff' * 50
            self.ser.write(stop_cmd)
            time.sleep(0.1)  # Wait for the command to be sent and processed
            print("Data stream stopped.")
            super().close()  # Call parent's close method



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
