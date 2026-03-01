#!/usr/bin/env python3
"""
Robotiq FT 300 Force/Torque Sensor Reader with Real-time Visualization
- 3 force curves (Fx, Fy, Fz)
- 3 torque curves (Mx, My, Mz)
- Data logging to bag file
- Real-time plot updates
"""

import time
import sys
import argparse
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

from robokit.data_manager.ft300_handler import FT300Handler as FT300


class FT300Visualizer:
    def __init__(self, window_size=500, save_data=True):
        """
        Initialize the visualizer

        Args:
            window_size: Number of data points to display in the plot
            save_data: Whether to save data to a bag file
        """
        self.window_size = window_size
        self.save_data = save_data

        # Data buffers using deque for efficient append/pop operations
        self.time_data = deque(maxlen=window_size)
        self.fx_data = deque(maxlen=window_size)
        self.fy_data = deque(maxlen=window_size)
        self.fz_data = deque(maxlen=window_size)
        self.mx_data = deque(maxlen=window_size)
        self.my_data = deque(maxlen=window_size)
        self.mz_data = deque(maxlen=window_size)

        # Full data storage for bag file
        if self.save_data:
            self.full_data = {
                'timestamps': [],
                'forces': [],  # [Fx, Fy, Fz]
                'torques': []  # [Mx, My, Mz]
            }

        self.start_time = None

        # Set up the plot
        self.setup_plot()

    def setup_plot(self):
        """Initialize matplotlib figure and axes"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('FT300 Sensor Real-time Data', fontsize=16, fontweight='bold')

        # Force subplot
        self.line_fx, = self.ax1.plot([], [], 'r-', linewidth=2, label='Fx')
        self.line_fy, = self.ax1.plot([], [], 'g-', linewidth=2, label='Fy')
        self.line_fz, = self.ax1.plot([], [], 'b-', linewidth=2, label='Fz')
        self.ax1.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
        self.ax1.set_title('Force Components', fontsize=14)
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)

        # Torque subplot
        self.line_mx, = self.ax2.plot([], [], 'r-', linewidth=2, label='Mx')
        self.line_my, = self.ax2.plot([], [], 'g-', linewidth=2, label='My')
        self.line_mz, = self.ax2.plot([], [], 'b-', linewidth=2, label='Mz')
        self.ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Torque (Nm)', fontsize=12, fontweight='bold')
        self.ax2.set_title('Torque Components', fontsize=14)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    def update_data(self, ft_data):
        """
        Update data buffers with new sensor reading

        Args:
            ft_data: List of 6 values [Fx, Fy, Fz, Mx, My, Mz]
        """
        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time() - self.start_time

        # Update deque buffers for plotting
        self.time_data.append(current_time)
        self.fx_data.append(ft_data[0])
        self.fy_data.append(ft_data[1])
        self.fz_data.append(ft_data[2])
        self.mx_data.append(ft_data[3])
        self.my_data.append(ft_data[4])
        self.mz_data.append(ft_data[5])

        # Save to full data storage
        if self.save_data:
            self.full_data['timestamps'].append(current_time)
            self.full_data['forces'].append([ft_data[0], ft_data[1], ft_data[2]])
            self.full_data['torques'].append([ft_data[3], ft_data[4], ft_data[5]])

    def update_plot(self, frame):
        """Update plot with current data (called by FuncAnimation)"""
        if len(self.time_data) == 0:
            return self.line_fx, self.line_fy, self.line_fz, self.line_mx, self.line_my, self.line_mz

        time_array = np.array(self.time_data)

        # Update force lines
        self.line_fx.set_data(time_array, np.array(self.fx_data))
        self.line_fy.set_data(time_array, np.array(self.fy_data))
        self.line_fz.set_data(time_array, np.array(self.fz_data))

        # Update torque lines
        self.line_mx.set_data(time_array, np.array(self.mx_data))
        self.line_my.set_data(time_array, np.array(self.my_data))
        self.line_mz.set_data(time_array, np.array(self.mz_data))

        # Auto-scale axes
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        return self.line_fx, self.line_fy, self.line_fz, self.line_mx, self.line_my, self.line_mz

    def save_bag(self, filename=None):
        """Save collected data to a pickle file (bag)"""
        if not self.save_data:
            print("Data saving was disabled")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ft300_data_{timestamp}.bag"

        with open(filename, 'wb') as f:
            pickle.dump(self.full_data, f)

        print(f"\nData saved to: {filename}")
        print(f"Total samples: {len(self.full_data['timestamps'])}")
        if len(self.full_data['timestamps']) > 0:
            duration = self.full_data['timestamps'][-1] - self.full_data['timestamps'][0]
            print(f"Duration: {duration:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time FT300 sensor visualization with data logging'
    )
    parser.add_argument('--port_prefix', default='/dev/ttyUSB',
                        help='Serial port prefix')
    parser.add_argument('--port_id', type=int, default=0,
                        help='Serial port ID')
    parser.add_argument('--window_size', type=int, default=500,
                        help='Number of data points to display')
    parser.add_argument('--no_save', action='store_true',
                        help='Disable data logging to bag file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output bag filename (default: auto-generated)')
    args = parser.parse_args()

    # Initialize sensor
    print("Connecting to FT300 sensor...")
    sensor = FT300(
        port_prefix=args.port_prefix,
        port_id=args.port_id,
    )

    # Connection retry loop
    for attempt in range(5):
        if sensor.connect():
            print("✓ Sensor connected successfully!")
            break
        print(f"✗ Retrying connection... ({attempt + 1}/5)")
        time.sleep(1)
    else:
        print("✗ Failed to connect after 5 attempts")
        sys.exit(1)

    # Initialize visualizer
    visualizer = FT300Visualizer(
        window_size=args.window_size,
        save_data=not args.no_save
    )

    print("\n" + "=" * 70)
    print("Real-time Force/Torque Visualization")
    print("=" * 70)
    print("Press Ctrl+C to stop and save data")
    print("=" * 70 + "\n")

    # Data reading state
    read_count = 0
    error_count = 0
    last_valid_data = None

    # def data_generator():
    #     """Generator function to read sensor data"""
    #     nonlocal read_count, error_count, last_valid_data
    #
    #     while True:
    #         # Read latest data from sensor
    #         data = sensor.read_ft()
    #
    #         if data:
    #             visualizer.update_data(data)
    #             last_valid_data = data
    #             read_count += 1
    #             error_count = 0
    #
    #             # Print periodic status
    #             if read_count % 100 == 0:
    #                 print(f"[{read_count:6d} samples] Last: "
    #                       f"F=({data[0]:6.3f}, {data[1]:6.3f}, {data[2]:6.3f}) N  "
    #                       f"M=({data[3]:6.4f}, {data[4]:6.4f}, {data[5]:6.4f}) Nm")
    #         else:
    #             error_count += 1
    #             if error_count % 100 == 0:
    #                 print(f"[Warning] No valid data received (errors: {error_count})")
    #
    #         yield
    #         time.sleep(0.001)  # ~1000Hz attempt rate for latest data
    def data_generator():
        """Generator function to read sensor data"""
        nonlocal read_count, error_count, last_valid_data

        while True:
            # 🔥 使用实时模式读取
            # data = sensor.read_ft_realtime()  # 或 sensor.read_ft(flush_old=True)
            data = sensor.read_ft()

            if data:
                visualizer.update_data(data)
                last_valid_data = data
                read_count += 1
                error_count = 0

                if read_count % 100 == 0:
                    print(f"[{read_count:6d} samples] Last: "
                          f"F=({data[0]:6.3f}, {data[1]:6.3f}, {data[2]:6.3f}) N  "
                          f"M=({data[3]:6.4f}, {data[4]:6.4f}, {data[5]:6.4f}) Nm")
            else:
                error_count += 1

            yield
            time.sleep(0.001)  # 1ms 轮询间隔

    # Start animation
    gen = data_generator()

    try:
        # Create animation with data generator
        anim = FuncAnimation(
            visualizer.fig,
            visualizer.update_plot,
            init_func=lambda: (visualizer.line_fx, visualizer.line_fy,
                               visualizer.line_fz, visualizer.line_mx,
                               visualizer.line_my, visualizer.line_mz),
            frames=gen,
            interval=50,  # 20 Hz plot update rate
            blit=True,
            cache_frame_data=False
        )

        plt.show()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Stopping data collection...")
        print("=" * 70)
        print(f"✓ Successfully read {read_count} packets")

        if last_valid_data:
            print(f"\nLast reading:")
            print(f"  Forces (N):  Fx={last_valid_data[0]:8.6f}, "
                  f"Fy={last_valid_data[1]:8.6f}, Fz={last_valid_data[2]:8.6f}")
            print(f"  Torques (Nm): Mx={last_valid_data[3]:7.6f}, "
                  f"My={last_valid_data[4]:7.6f}, Mz={last_valid_data[5]:7.6f}")

    finally:
        # Save data and cleanup
        if not args.no_save:
            visualizer.save_bag(args.output)

        sensor.close()
        print("\n✓ Sensor disconnected")
        print("=" * 70 + "\n")


if __name__ == '__main__':
    main()