import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def concatenate_rgb_images(img1, img2, vertical=False, smaller_size=2):
    """
    拼接两个RGB图像（左右拼接）

    参数：
    img1 (np.ndarray): 第一个RGB图像，形状为 (height, width, 3)
    img2 (np.ndarray): 第二个RGB图像，形状为 (height, width, 3)

    返回：
    np.ndarray: 拼接后的图像，形状为 (height, width1 + width2, 3)
    """
    height = img1.shape[0]
    width = img1.shape[1]
    img1 = np.array(Image.fromarray(img1).resize((width // smaller_size, height // smaller_size)))
    hw_ratio2 = float(img2.shape[0]) / float(img2.shape[1])

    if not vertical:
        # 确保两个图像的高度相同
        img2 = np.array(Image.fromarray(img2).resize((int(height / hw_ratio2) // smaller_size, height // smaller_size)))
        # 使用numpy的hstack来拼接两个图像
        return np.hstack((img1, img2))
    else:
        # 确保两个图像的宽度相同
        img2 = np.array(Image.fromarray(img2).resize((width // smaller_size, int(width * hw_ratio2) // smaller_size)))
        # 使用numpy的vstack来拼接两个图像
        return np.vstack((img1, img2))


def plot_action_wrt_time(action_data: np.ndarray):
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 初始化曲线
    frames_cnt = action_data.shape[0]
    lines = [ax.plot([], [], label=label)[0] for label in ['x', 'y', 'z', 'a', 'b', 'c', 'g']]
    ax.set_xlim(0, frames_cnt)
    ax.set_ylim(np.min(action_data), np.max(action_data))
    ax.set_xlabel('Time (Frames)')
    ax.set_ylabel('Value')
    ax.legend()

    # 用于保存每一帧的图像列表
    frames = []

    # 手动绘制每一帧并将其保存到列表
    for frame_num in range(frames_cnt):
        # 更新曲线数据
        for i, line in enumerate(lines):
            line.set_data(np.arange(frame_num), action_data[:frame_num, i])  # 更新数据

        plt.tight_layout()

        # 将当前帧保存到帧列表中
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
        frames.append(img)  # 将图像添加到帧列表中

    plt.close(fig)
    return frames, fig, ax