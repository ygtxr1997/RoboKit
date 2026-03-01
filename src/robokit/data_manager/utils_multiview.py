from typing import Union, Tuple
import numpy as np
import torch


def get_frames_from_multiview_video(video_B_VT_H_W_C: np.ndarray, sample_n_views: int, start_idx=0, end_idx=None):
    # video_B_VT_H_W_C: (B,V*T,H,W,C), in [0,255]
    B = video_B_VT_H_W_C.shape[0]
    T = video_B_VT_H_W_C.shape[1] // sample_n_views
    V = sample_n_views
    assert video_B_VT_H_W_C.shape[1] == T * V, \
        f"Expected first dimension to be divisible by {V}, got {video_B_VT_H_W_C.shape[1]}"
    if end_idx is None:
        end_idx = T

    view_starts = np.arange(V) * T  # [0, T, 2T, ..., (V-1)*T]
    time_offsets = np.arange(start_idx, end_idx)  # [start_idx, start_idx+1, ..., end_idx-1]
    select_indices = (view_starts[:, None] + time_offsets).flatten()  # (V*(end_idx-start_idx),)

    select_frames_B_VT_H_W_C = video_B_VT_H_W_C[:, select_indices]  # (B, V*(end_idx-start_idx), H, W, C)
    return select_frames_B_VT_H_W_C


def cat_multiview_video_with_zeros(video_B_VT_H_W_C: np.ndarray, sample_n_views: int, zero_length: int):
    V = sample_n_views
    B = video_B_VT_H_W_C.shape[0]
    total_frames = video_B_VT_H_W_C.shape[1]
    T = total_frames // V
    H, W, C = video_B_VT_H_W_C.shape[2:]

    assert total_frames == T * V, f"Expected first dimension to be divisible by {V}, got {total_frames}"

    video_reshaped = video_B_VT_H_W_C.reshape(B, V, T, H, W, C)
    zeros = np.zeros((B, V, zero_length, H, W, C), dtype=np.uint8)

    # Concatenate along the time dimension: (V, T+zero_length, H, W, C)
    video_with_zeros = np.concatenate([video_reshaped, zeros], axis=2)

    # Reshape back: (V, T+zero_length, H, W, C) -> (V*(T+zero_length), H, W, C)
    result = video_with_zeros.reshape(B, V * (T + zero_length), H, W, C)
    return result


def cat_multiview_video_with_another(
        video_A_B_VT_H_W_C: np.ndarray,
        video_B_B_VT_H_W_C: np.ndarray,
        sample_n_views: int,
):
    V = sample_n_views
    B = video_A_B_VT_H_W_C.shape[0]
    total_frames_A = video_A_B_VT_H_W_C.shape[1]
    total_frames_B = video_B_B_VT_H_W_C.shape[1]
    T_A = total_frames_A // V
    T_B = total_frames_B // V
    H, W, C = video_A_B_VT_H_W_C.shape[2:]

    assert total_frames_A == T_A * V, f"Expected first dimension to be divisible by {V}, got {total_frames_A}"
    assert total_frames_B == T_B * V, f"Expected first dimension to be divisible by {V}, got {total_frames_B}"
    assert video_A_B_VT_H_W_C.shape[0] == video_B_B_VT_H_W_C.shape[0], \
        f"Shape mismatch: {video_A_B_VT_H_W_C.shape[0]} vs {video_B_B_VT_H_W_C.shape[0]}"
    assert video_A_B_VT_H_W_C.shape[2:] == video_B_B_VT_H_W_C.shape[2:], \
        f"Shape mismatch: {video_A_B_VT_H_W_C.shape[2:]} vs {video_B_B_VT_H_W_C.shape[2:]}"
    # Reshape to (B, V, T, H, W, C)
    video_A_reshaped = video_A_B_VT_H_W_C.reshape(B, V, T_A, H, W, C)
    video_B_reshaped = video_B_B_VT_H_W_C.reshape(B, V, T_B, H, W, C)
    # Concatenate along the time dimension: (B, V, T_A+T_B, H, W, C)
    concatenated = np.concatenate([video_A_reshaped, video_B_reshaped], axis=2)
    # Reshape back to (B, V*(T_A+T_B), H, W, C)
    result = concatenated.reshape(B, V * (T_A + T_B), H, W, C)
    return result


def replace_multiview_video_back_with_another(
        ori_video_B_C_VT_H_W: torch.Tensor,
        new_video_B_C_VT_H_W: torch.Tensor,
        sample_n_views: int,
        replace_length: int,
):
    B, C, VT_ori, H, W = ori_video_B_C_VT_H_W.shape
    VT_new = new_video_B_C_VT_H_W.shape[2]
    T_ori = VT_ori // sample_n_views
    T_new = VT_new // sample_n_views
    V = sample_n_views

    assert VT_ori == T_ori * V, \
        f"Expected third dimension to be divisible by {V}, got {VT_ori}"
    assert VT_new == T_new * V, \
        f"Expected third dimension to be divisible by {V}, got {VT_new}"
    assert replace_length <= T_ori and replace_length <= T_new, \
        f"replace_length {replace_length} exceeds original or new video length"
    assert ori_video_B_C_VT_H_W.shape[0] == new_video_B_C_VT_H_W.shape[0], \
        f"Shape mismatch: {ori_video_B_C_VT_H_W.shape[0]} vs {new_video_B_C_VT_H_W.shape[0]}"

    # Reshape to (B, C, V, T, H, W)
    ori_reshaped = ori_video_B_C_VT_H_W.view(B, C, V, T_ori, H, W)
    new_reshaped = new_video_B_C_VT_H_W.view(B, C, V, T_new, H, W)

    # Replace the last `replace_length` frames along the time dimension
    ori_reshaped[:, :, :, -replace_length:, :, :] = new_reshaped[:, :, :, -replace_length:, :, :]

    # Reshape back to (B, C, V*T, H, W)
    result = ori_reshaped.view(B, C, VT_ori, H, W)
    return result

