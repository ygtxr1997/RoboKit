import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyrealsense2 import hole_filling_filter
from tqdm import tqdm
import json
import cv2

import pyrealsense2 as rs


class RealsenseHandler(object):
    def __init__(self, img_width: int = 848, img_height: int = 480, frame_rate: int = 60):
        # 确定图像的输入分辨率与帧率
        self.img_width = img_width
        self.img_height = img_height
        self.frame_rate = frame_rate

        # 注册数据流，并对其图像
        self.align = rs.align(rs.stream.color)
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, img_width, img_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.color, img_width, img_height, rs.format.bgr8, frame_rate)

        # check相机是不是进来了
        connect_devices = []
        for d in rs.context().devices:
            print('[RealsenseHandler] Found device: ',
                  d.get_info(rs.camera_info.name), ' ',
                  d.get_info(rs.camera_info.serial_number))
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                connect_devices.append(d.get_info(rs.camera_info.serial_number))
        print("[RealsenseHandler] connected devices:", connect_devices)

        pipeline_profiles = []
        pipelines = []
        for idx, connect_device in enumerate(connect_devices):
            # 确认相机并获取相机的内部参数
            pipeline = rs.pipeline()
            rs_config.enable_device(connect_device)
            pipeline_profile = pipeline.start(rs_config)

            pipeline_profiles.append(pipeline_profile)
            pipelines.append(pipeline)

        self.rs_config = rs_config
        self.connect_devices = connect_devices
        self.pipeline_profiles = pipeline_profiles
        self.pipelines = pipelines

        self.auto_just_wb = True
        self.set_ae_wb_auto(auto_adjust=True)
        print(f"[RealsenseHandler] Initialized finished. WxH={self.img_width}x{self.img_height} fps={self.frame_rate}.")

    def get_advance_mode(self):
        DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03",
                           "0B07", "0B3A", "0B5C", "0B5B"]

        def find_device_that_supports_advanced_mode():
            ctx = rs.context()
            ds5_dev = rs.device()
            devices = ctx.query_devices()
            print("Find devices supports advanced mode:", devices)
            for dev in devices:
                if dev.supports(rs.camera_info.product_id) and str(
                        dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
                    if dev.supports(rs.camera_info.name):
                        print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                    return dev
            raise Exception("No D400 product line device that supports advanced mode was found")

        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        return advnc_mode

    def dump_current_control_to_json(self):
        advnc_mode = self.get_advance_mode()
        # print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
        #
        # # Get each control's current value
        # print("Depth Control: \n", advnc_mode.get_depth_control())
        # print("RSM: \n", advnc_mode.get_rsm())
        # print("RAU Support Vector Control: \n", advnc_mode.get_rau_support_vector_control())
        # print("Color Control: \n", advnc_mode.get_color_control())
        # print("RAU Thresholds Control: \n", advnc_mode.get_rau_thresholds_control())
        # print("SLO Color Thresholds Control: \n", advnc_mode.get_slo_color_thresholds_control())
        # print("SLO Penalty Control: \n", advnc_mode.get_slo_penalty_control())
        # print("HDAD: \n", advnc_mode.get_hdad())
        # print("Color Correction: \n", advnc_mode.get_color_correction())
        # print("Depth Table: \n", advnc_mode.get_depth_table())
        # print("Auto Exposure Control: \n", advnc_mode.get_ae_control())
        # print("Census: \n", advnc_mode.get_census())

        # Serialize all controls to a Json string
        serialized_string = advnc_mode.serialize_json()
        # print("Controls as JSON: \n", serialized_string)
        as_json_object = json.loads(serialized_string)
        # 保存 JSON 对象到文件
        with open('tmp_controls.json', 'w') as json_file:
            json.dump(as_json_object, json_file, indent=4)  # indent=4 用于格式化输出，便于阅读
        print("JSON data has been saved to 'controls.json'.")

    def load_control_from_json(self):
        advnc_mode = self.get_advance_mode()
        with open('tmp_controls.json', 'r') as json_file:
            as_json_object = json.load(json_file)
        # We can also load controls from a json string
        # For Python 2, the values in 'as_json_object' dict need to be converted from unicode object to utf-8
        if type(next(iter(as_json_object))) != str:
            as_json_object = {k.encode('utf-8'): v.encode("utf-8") for k, v in as_json_object.items()}
        # The C++ JSON parser requires double-quotes for the json object so we need
        # to replace the single quote of the pythonic json to double-quotes
        json_string = str(as_json_object).replace("'", '\"')
        advnc_mode.load_json(json_string)
        print("Camera advanced loaded from 'controls.json'.")

    def set_ae_wb(self, camera_idx: int = 0, hue=-1., wb=3200, exposure=100, saturation=64):
        profile = self.pipeline_profiles[camera_idx]

        device = profile.get_device()
        color_sensor = None
        for sensor in device.query_sensors():
            if "rgb" in sensor.get_info(rs.camera_info.name).lower():
                color_sensor = sensor
                break
        if not color_sensor:
            raise Exception("No color sensor found")

        now_hue = color_sensor.get_option(rs.option.hue)
        now_wb = color_sensor.get_option(rs.option.white_balance)
        now_exposure = color_sensor.get_option(rs.option.exposure)
        now_gamma = color_sensor.get_option(rs.option.gamma)
        now_saturation = color_sensor.get_option(rs.option.saturation)
        print(now_hue, now_wb, now_exposure, now_gamma, now_saturation)

        color_sensor.set_option(rs.option.white_balance, wb)  # 1000:cool, 7000:warm
        color_sensor.set_option(rs.option.hue, hue)  # -10:red, +10:green
        color_sensor.set_option(rs.option.exposure, exposure)
        color_sensor.set_option(rs.option.saturation, saturation)

    def set_ae_wb_auto(self, auto_adjust: bool = True, camera_idx: int = -1):
        if self.auto_just_wb == auto_adjust:
            return

        if camera_idx == -1:
            profiles = self.pipeline_profiles
        else:
            profiles = [self.pipeline_profiles[camera_idx]]

        for profile in profiles:
            device = profile.get_device()
            color_sensor = None
            for sensor in device.query_sensors():
                if "rgb" in sensor.get_info(rs.camera_info.name).lower():
                    color_sensor = sensor
                    break
            if not color_sensor:
                raise Exception("No color sensor found")

            color_sensor.set_option(rs.option.enable_auto_white_balance, auto_adjust)
            color_sensor.set_option(rs.option.enable_auto_exposure, auto_adjust)

        self.auto_just_wb = auto_adjust
        camera_info = "all" if camera_idx == -1 else [str(camera_idx)]
        print(f"[RealsenseHandler] Switching auto-white-balance mode to: {auto_adjust}, cameras: {camera_info}")

    def capture_frames(self, save_prefix: str = None, skip_frames: int = 0, skip_seconds: float = -1.):
        pipelines = self.pipelines

        pipeline1 = pipelines[0]
        pipeline2 = pipelines[1]

        if skip_frames > 0:
            for i in tqdm(range(skip_frames), desc="[RealsenseHandler] Skipping some frames"):
                pipeline1.wait_for_frames()
                pipeline2.wait_for_frames()

        if skip_seconds > 0:
            print(f"[RealsenseHandler] Skipping frames by waiting for {skip_seconds:.3f} seconds.")
            time.sleep(skip_seconds)

        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()

        aligned_frames1 = self.align.process(frames1)
        aligned_frames2 = self.align.process(frames2)

        # 将对其的RGB—D图取出来
        depth_frame1 = frames1.get_depth_frame()
        color_frame1 = frames1.get_color_frame()
        depth_frame2 = frames2.get_depth_frame()
        color_frame2 = frames2.get_color_frame()

        # 将图像转换为numpy数组
        color_image1 = np.asanyarray(color_frame1.get_data())  # BGR
        depth_image1 = np.asanyarray(depth_frame1.get_data())  # in [0,65535]
        color_image2 = np.asanyarray(color_frame2.get_data())  # BGR
        depth_image2 = np.asanyarray(depth_frame2.get_data())  # in [0,65535]

        def map_depth_with_color(depth_image: np.ndarray) -> Image.Image:
            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGRA2RGBA)
            depth_colored_image = Image.fromarray(depth_colored)
            return depth_colored_image

        depth_colored_image1 = map_depth_with_color(depth_image1)
        depth_colored_image2 = map_depth_with_color(depth_image2)

        color_image1 = color_image1[:, :, ::-1]  # BGR to RGB
        color_image2 = color_image2[:, :, ::-1]

        if save_prefix is not None:
            image_filename = f"{save_prefix}_1_color.jpg"
            depth_image_filename = f"{save_prefix}_1_depth.png"
            Image.fromarray(color_image1).save(image_filename)
            Image.fromarray(depth_image1).save(depth_image_filename)

            image_filename = f"{save_prefix}_2_color.jpg"
            depth_image_filename = f"{save_prefix}_2_depth.png"
            Image.fromarray(color_image2).save(image_filename)
            Image.fromarray(depth_image2).save(depth_image_filename)

            depth_colored_image1.save(f"{save_prefix}_1_depth_color.png")
            depth_colored_image2.save(f"{save_prefix}_2_depth_color.png")

            print(f"[RealsenseHandler] Images saved as: {save_prefix}_xxx.jpg/png")

        return {
            "color1": color_image1,
            "color2": color_image2,
            "depth1": depth_image1,
            "depth2": depth_image2,
        }

    def stop(self):
        for pipeline in self.pipelines:
            pipeline.stop()
        print("Stopping all pipelines")
