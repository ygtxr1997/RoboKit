# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_episode_stats


def sample_episode_video_frames(dataset: LeRobotDataset, episode_index: int, ft_key: str) -> np.ndarray:
    ep_len = dataset.meta.episodes[episode_index]["length"]
    sampled_indices = sample_indices(ep_len)
    query_timestamps = dataset._get_query_timestamps(0.0, {ft_key: sampled_indices})
    video_frames = dataset._query_videos(query_timestamps, episode_index)
    return video_frames[ft_key].numpy()


def convert_episode_stats_old(dataset: LeRobotDataset, ep_idx: int):
    ep_start_idx = dataset.episode_data_index["from"][ep_idx]
    ep_end_idx = dataset.episode_data_index["to"][ep_idx]
    ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    ep_stats = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            # We sample only for videos
            ep_ft_data = sample_episode_video_frames(dataset, ep_idx, key)
        else:
            ep_ft_data = np.array(ep_data[key])

        axes_to_reduce = (0, 2, 3) if ft["dtype"] in ["image", "video"] else 0
        keepdims = True if ft["dtype"] in ["image", "video"] else ep_ft_data.ndim == 1
        ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)

        if ft["dtype"] in ["image", "video"]:  # remove batch dim
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()
            }

    dataset.meta.episodes_stats[ep_idx] = ep_stats


def convert_episode_stats(dataset: LeRobotDataset, ep_idx: int):
    ep_start_idx = dataset.episode_data_index["from"][ep_idx]
    ep_end_idx = dataset.episode_data_index["to"][ep_idx]

    # [修改点 0] 使用 dataset.__getitem__ 获取原始数据，而不是 dataset.hf_dataset.select
    # 这样可以利用 Dataset 类可能存在的 decode 逻辑，并且 select 可能会很慢
    # ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    ep_stats = {}
    for key, ft in dataset.features.items():
        # [修改点 1] 获取数据并标准化为 numpy array
        if ft["dtype"] == "video":
            # We sample only for videos (这里返回的是 (N, C, H, W) 或 (N, H, W, C)，取决于实现)
            ep_ft_data = sample_episode_video_frames(dataset, ep_idx, key)
        else:
            # 直接从 HF dataset 读取
            # ep_ft_data = np.array(ep_data[key])
            # 为了更好的兼容性，直接读这一列
            ep_ft_data = np.array(dataset.hf_dataset[range(ep_start_idx, ep_end_idx)][key])

        # [修改点 2] 关键修复：处理图像/视频数据的维度
        if ft["dtype"] in ["image", "video"]:
            # 如果是 PyTorch Tensor，转为 numpy
            if hasattr(ep_ft_data, "numpy"):
                ep_ft_data = ep_ft_data.numpy()

            # 1. 检查是否只有 3 维 (N, H, W) 或 (N, C, H*W?) -> 补全为 4 维
            if ep_ft_data.ndim == 3:
                # 假设是 (N, H, W) -> 变成 (N, 1, H, W) 以适配下面的 axis reduce
                # 或者如果原来的逻辑假设是 (N, C, H, W)，则插入 dim 1
                # 原代码 axes_to_reduce = (0, 2, 3)，这几乎肯定是针对 (N, C, H, W) 格式的
                # 即 Batch, Channel, Height, Width ->Reduce N, H, W -> 剩 Channel
                ep_ft_data = ep_ft_data[:, None, :, :]

            # 2. 检查是否是 (N, H, W, C) 格式 (Numpy 通常格式)
            # 如果是 (N, H, W, C)，且 axes=(0,2,3)，那会把 Width 当做 Channel 保留，把 Channel 当做 Width reduce 掉，这是错的。
            # 假设 C 通常很小 (<=4), H, W 很大 (>4)
            # 如果 shape 是 (N, 224, 224, 3)
            if ep_ft_data.shape[1] > 4 and ep_ft_data.shape[-1] <= 4:
                # 这是一个 (N, H, W, C) 或者 (N, H, W, 1) 的数组
                # 我们需要把它变成 (N, C, H, W) 才能适配 axis=(0, 2, 3)
                ep_ft_data = ep_ft_data.transpose(0, 3, 1, 2)

        # [修改点 3] 确认 Axis
        # LeRobot 0.3.3 通常假设数据是 (N, C, H, W)
        # axes=(0, 2, 3) 意味着沿着 Batch, Height, Width 进行压缩，保留 Channel
        if ft["dtype"] in ["image", "video"]:
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            axes_to_reduce = 0
            # 这里原本的 keepdims 逻辑有点奇怪，如果是 (N, D)，keepdims=True 会变成 (1, D)
            # 保持原样即可
            keepdims = ep_ft_data.ndim == 1

        try:
            ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)
        except np.exceptions.AxisError:
            # Fallback for weird shapes
            print(f"[WARN] AxisError for key {key}, shape {ep_ft_data.shape}. Reducing axis 0 only.")
            ep_stats[key] = get_feature_stats(ep_ft_data, axis=0, keepdims=keepdims)

        if ft["dtype"] in ["image", "video"]:  # remove batch dim
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()
            }

    dataset.meta.episodes_stats[ep_idx] = ep_stats



def convert_stats(dataset: LeRobotDataset, num_workers: int = 0):
    assert dataset.episodes is None
    print("Computing episodes stats")
    total_episodes = dataset.meta.total_episodes
    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(convert_episode_stats, dataset, ep_idx): ep_idx
                for ep_idx in range(total_episodes)
            }
            for future in tqdm(as_completed(futures), total=total_episodes):
                future.result()
    else:
        for ep_idx in tqdm(range(total_episodes)):
            convert_episode_stats(dataset, ep_idx)

    for ep_idx in tqdm(range(total_episodes)):
        write_episode_stats(ep_idx, dataset.meta.episodes_stats[ep_idx], dataset.root)


def check_aggregate_stats(
    dataset: LeRobotDataset,
    reference_stats: dict[str, dict[str, np.ndarray]],
    video_rtol_atol: tuple[float] = (1e-2, 1e-2),
    default_rtol_atol: tuple[float] = (5e-6, 6e-5),
):
    """Verifies that the aggregated stats from episodes_stats are close to reference stats."""
    agg_stats = aggregate_stats(list(dataset.meta.episodes_stats.values()))
    for key, ft in dataset.features.items():
        # These values might need some fine-tuning
        if ft["dtype"] == "video":
            # to account for image sub-sampling
            rtol, atol = video_rtol_atol
        else:
            rtol, atol = default_rtol_atol

        for stat, val in agg_stats[key].items():
            if key in reference_stats and stat in reference_stats[key]:
                err_msg = f"feature='{key}' stats='{stat}'"
                np.testing.assert_allclose(
                    val, reference_stats[key][stat], rtol=rtol, atol=atol, err_msg=err_msg
                )
