import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

profile = pipeline.start(config)
pipeline.stop()
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


