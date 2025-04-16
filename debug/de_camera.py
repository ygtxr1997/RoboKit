import pyrealsense2 as rs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def set_ae_awb(profile):
    device = profile.get_device()
    color_sensor = None
    for sensor in device.query_sensors():
        if "rgb" in sensor.get_info(rs.camera_info.name).lower():
            color_sensor = sensor
            break
    if not color_sensor:
        raise Exception("No color sensor found")

    color_sensor.set_option(rs.option.white_balance, 3600)
    color_sensor.set_option(rs.option.exposure, 100)
    print("Color sensor setting: WB=3600, Exposure=100")


# 确定图像的输入分辨率与帧率
resolution_width = 1280  # pixels, 640
resolution_height = 720  # pixels, 480
frame_rate = 15  # fps, 15

# 注册数据流，并对其图像
align = rs.align(rs.stream.color)
rs_config = rs.config()
rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
### d435i
#
# rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
# rs_config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, frame_rate)
# check相机是不是进来了
connect_devices = []
for d in rs.context().devices:
    print('Found device: ',
          d.get_info(rs.camera_info.name), ' ',
          d.get_info(rs.camera_info.serial_number))
    if d.get_info(rs.camera_info.name).lower() != 'platform camera':
        connect_devices.append(d.get_info(rs.camera_info.serial_number))

print(connect_devices)

pipelines = []
for connect_device in connect_devices:
    # 确认相机并获取相机的内部参数
    pipeline = rs.pipeline()
    rs_config.enable_device(connect_device)
    pipeline_profile = pipeline.start(rs_config)
    set_ae_awb(pipeline_profile)

    pipelines.append(pipeline)

pipeline = pipelines[0]
for _ in range(100):
    frames = pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

# 将图像转换为numpy数组
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())
image_filename = f"exp_1.jpg"
depth_image_filename = f"exp_1_depth.png"

Image.fromarray(color_image).save(image_filename)
Image.fromarray(depth_image).save(depth_image_filename)


for pipeline in pipelines:
    pipeline.stop()

# 使用Matplotlib的colormap进行颜色映射
depth_normalized = np.uint8((depth_image / np.max(depth_image)) * 255)
colormap = plt.get_cmap('plasma')  # 使用Plasma colormap
depth_colored = colormap(depth_normalized / 255)

# 将RGB+Alpha转换为仅RGB
depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

# 使用Pillow保存彩色图像
depth_colored_image = Image.fromarray(depth_colored)
depth_colored_image.save("depth_colored_image.png")
depth_colored_image.show()

print("OK")

# pipe = rs.pipeline()
# profile = pipe.start()
# try:
#   for i in range(0, 100):
#     frames = pipe.wait_for_frames()
#     for f in frames:
#       print(f.profile)
#   print("OK")
# finally:
#     pipe.stop()


# import cv2


# def list_available_cameras():
#     # 检测从索引 0 到 9 的摄像头设备
#     available_cameras = []
#     for i in range(10):  # 一般来说，系统的摄像头设备的索引不会超过 9
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():  # 如果摄像头设备打开成功
#             available_cameras.append(i)
#             cap.release()  # 关闭摄像头设备
#     return available_cameras

# # 获取所有可用的摄像头设备索引
# cameras = list_available_cameras()
# if cameras:
#     print(f"可用的摄像头设备索引: {cameras}")
# else:
#     print("没有找到可用的摄像头设备。")


# import cv2

# camera_index = 0  # 默认使用第一个摄像头设备

# cap = cv2.VideoCapture(camera_index)

# if not cap.isOpened():
#     print("无法打开摄像头")
# else:
#     print(f"正在显示摄像头 {camera_index} 的视频流...")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("无法从摄像头读取数据")
#             break

#         # 显示视频流
#         cv2.imshow('Camera Stream', frame)

#         # 按 'q' 键退出视频流显示
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# import cv2

# # 尝试打开设备索引为0~5的摄像头
# for i in range(0, 5):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"打开摄像头索引 {i} 成功")
#         break

# if not cap.isOpened():
#     print("找不到可用的摄像头")
# else:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("无法读取图像")
#             break
#         cv2.imshow("RGB Image", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


