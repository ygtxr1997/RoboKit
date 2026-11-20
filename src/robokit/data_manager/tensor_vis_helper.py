import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import h5py
import numpy as np
import torch


TemporalTensor = Union[torch.Tensor, np.ndarray]
FilePath = Union[str, os.PathLike]


class TensorVisHelper:
    """Collect and persist named batched temporal tensors for visualization."""

    FILE_FORMAT = "robokit.tensor_vis"
    FORMAT_VERSION = 1

    def __init__(self, initial_time_capacity: int = 128):
        if initial_time_capacity <= 0:
            raise ValueError("initial_time_capacity should be greater than 0")

        self.initial_time_capacity = initial_time_capacity
        self._buffers: Dict[str, np.ndarray] = {}
        self._batch_size: Optional[int] = None
        self._time_length = 0
        self._time_capacity = 0

    def add_temporal_tensor(self, tensor_dict: Mapping[str, TemporalTensor]) -> None:
        """Append temporal tensor chunks with shape ``(B, chunk_T, D)``.

        All tensors in one call must share B and chunk_T. The first call probes
        the tensor shapes and allocates buffers; later calls append along T.
        """
        if not isinstance(tensor_dict, Mapping) or not tensor_dict:
            raise ValueError("tensor_dict should be a non-empty mapping")

        arrays: Dict[str, np.ndarray] = {}
        batch_size = None
        chunk_time = None

        for key, tensor in tensor_dict.items():
            self._validate_key(key)
            x_B_T_D = self._to_numpy(tensor, key)
            if x_B_T_D.ndim != 3:
                raise ValueError(
                    f"tensor_dict['{key}'] should have shape (B, chunk_T, D), got {x_B_T_D.shape}"
                )
            if any(dim <= 0 for dim in x_B_T_D.shape):
                raise ValueError(
                    f"tensor_dict['{key}'] dimensions should be greater than 0, got {x_B_T_D.shape}"
                )

            current_batch_size, current_chunk_time, _ = x_B_T_D.shape
            if batch_size is None:
                batch_size = current_batch_size
                chunk_time = current_chunk_time
            elif current_batch_size != batch_size or current_chunk_time != chunk_time:
                raise ValueError(
                    "all tensors in one call should share the same B and chunk_T, "
                    f"got {(current_batch_size, current_chunk_time)} for '{key}' "
                    f"and expected {(batch_size, chunk_time)}"
                )

            arrays[key] = x_B_T_D

        if not self._buffers:
            self._initialize_buffers(arrays, batch_size, chunk_time)
        else:
            self._validate_existing_schema(arrays, batch_size)
            self._ensure_time_capacity(self._time_length + chunk_time)

        end_time = self._time_length + chunk_time
        for key, x_B_T_D in arrays.items():
            self._buffers[key][:, self._time_length:end_time, :] = x_B_T_D
        self._time_length = end_time
        print(f"[DEBUG][TensorVisHelper.add_temporal_tensor] bs={batch_size} chunk_T={chunk_time} ")

    def save(self, file_path: FilePath) -> List[str]:
        """Append buffered data to an HDF5 file and return new rollout paths."""
        if not self._buffers or self._time_length == 0:
            raise RuntimeError("no temporal tensor data to save")

        file_path = str(file_path).replace(" ", "_")
        save_path = Path(file_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        rollout_paths: List[str] = []
        with h5py.File(save_path, "a") as h5_file:
            self._initialize_or_validate_file(h5_file)
            rollouts_group = h5_file["rollouts"]
            next_rollout_index = int(h5_file.attrs["next_rollout_index"])
            created_group_names: List[str] = []

            try:
                for batch_index in range(self._batch_size):
                    while f"{next_rollout_index:08d}" in rollouts_group:
                        next_rollout_index += 1

                    group_name = f"{next_rollout_index:08d}"
                    rollout_group = rollouts_group.create_group(group_name)
                    created_group_names.append(group_name)
                    rollout_group.attrs["num_steps"] = self._time_length
                    tensors_group = rollout_group.create_group("tensors")

                    for key, buffer_B_T_D in self._buffers.items():
                        x_T_D = buffer_B_T_D[batch_index, :self._time_length, :]
                        tensors_group.create_dataset(key, data=x_T_D)

                    rollout_paths.append(f"/rollouts/{group_name}")
                    next_rollout_index += 1

                h5_file.attrs["next_rollout_index"] = next_rollout_index
                h5_file.flush()
            except Exception:
                for group_name in created_group_names:
                    if group_name in rollouts_group:
                        del rollouts_group[group_name]
                h5_file.flush()
                raise

        self._reset_buffers()
        return rollout_paths

    @classmethod
    def load(cls, file_path: FilePath) -> Dict[str, List[np.ndarray]]:
        """Load tensors grouped by key and rollout.

        The returned value for each key is a list of length B, where each item
        has shape ``(T_i, D)``. If every rollout has the same T, callers can use
        ``np.stack(values, axis=0)`` to obtain shape ``(B, T, D)``.
        """

        load_path = Path(file_path)
        if not load_path.exists():
            raise FileNotFoundError(f"tensor visualization file not found: {load_path}")

        with h5py.File(load_path, "r") as h5_file:
            cls._validate_file_header(h5_file)
            schema_group = h5_file["schema"]
            tensor_keys = sorted(schema_group.keys())
            loaded: Dict[str, List[np.ndarray]] = {key: [] for key in tensor_keys}

            rollouts_group = h5_file["rollouts"]
            for group_name in sorted(rollouts_group.keys()):
                rollout_group = rollouts_group[group_name]
                if "tensors" not in rollout_group:
                    raise ValueError(f"rollout '{group_name}' does not contain a tensors group")

                tensors_group = rollout_group["tensors"]
                if set(tensors_group.keys()) != set(tensor_keys):
                    raise ValueError(f"rollout '{group_name}' tensor keys do not match the file schema")

                for key in tensor_keys:
                    x_T_D = tensors_group[key][:]
                    expected_feature_dim = int(schema_group[key].attrs["feature_dim"])
                    expected_dtype = np.dtype(cls._decode_attr(schema_group[key].attrs["dtype"]))
                    if x_T_D.ndim != 2 or x_T_D.shape[1] != expected_feature_dim:
                        raise ValueError(
                            f"rollout '{group_name}' tensor '{key}' should have shape "
                            f"(T, {expected_feature_dim}), got {x_T_D.shape}"
                        )
                    if x_T_D.dtype != expected_dtype:
                        raise ValueError(
                            f"rollout '{group_name}' tensor '{key}' should have dtype "
                            f"{expected_dtype}, got {x_T_D.dtype}"
                        )
                    loaded[key].append(x_T_D)

        return loaded

    @classmethod
    def visualize(cls, file_path: FilePath) -> Dict[str, List[np.ndarray]]:
        """Load saved tensors for visualization and return them by key."""
        loaded = cls.load(file_path)

        # NOTE: Convert tensors that stack to (B, T, 1), (B, T, 2), or
        # (B, T, 3) into trajectory images.
        return loaded

    @classmethod
    def visualize2d(
            cls,
            file_path: FilePath,
            tensor_key: str,
            output_path: Optional[FilePath] = None,
            show: bool = True,
            rollout_indices: Optional[List[int]] = None,
            sample_every: int = 10,
            ) -> Tuple[object, object]:
        """Plot selected ``(T_i, 2)`` rollouts with a fixed time color range.

        Each trajectory changes from light to dark according to timestep in
        the fixed range [0, 300]. Green circles and red crosses mark trajectory
        starts and ends. ``rollout_indices`` uses zero-based indices; ``None``
        plots every rollout. ``sample_every`` controls the interval between
        sampled trajectory points and always preserves the final point. A
        constant-velocity Kalman filter smooths each trajectory before
        sampling. The returned objects are the matplotlib ``Figure`` and
        ``Axes``.
        """
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        loaded = cls.load(file_path)
        if tensor_key not in loaded:
            raise KeyError(
                f"tensor key '{tensor_key}' not found; available keys: {sorted(loaded)}"
            )

        trajectories = loaded[tensor_key]
        if not trajectories:
            raise ValueError(f"tensor key '{tensor_key}' does not contain any rollouts")
        if not isinstance(sample_every, int) or isinstance(sample_every, bool):
            raise TypeError("sample_every should be an integer")
        if sample_every <= 0:
            raise ValueError("sample_every should be greater than 0")

        if rollout_indices is None:
            selected_indices = list(range(len(trajectories)))
        else:
            if not rollout_indices:
                raise ValueError("rollout_indices should not be empty")
            if any(
                    not isinstance(index, int) or isinstance(index, bool)
                    for index in rollout_indices
                    ):
                raise TypeError("rollout_indices should contain only integers")
            if len(set(rollout_indices)) != len(rollout_indices):
                raise ValueError("rollout_indices should not contain duplicates")
            invalid_indices = [
                index for index in rollout_indices
                if index < 0 or index >= len(trajectories)
            ]
            if invalid_indices:
                raise IndexError(
                    f"rollout indices out of range: {invalid_indices}; "
                    f"available range is [0, {len(trajectories) - 1}]"
                )
            selected_indices = rollout_indices

        selected_trajectories = [
            (rollout_index, trajectories[rollout_index])
            for rollout_index in selected_indices
        ]
        for rollout_index, x_T_2 in selected_trajectories:
            if x_T_2.ndim != 2 or x_T_2.shape[1] != 2:
                raise ValueError(
                    f"rollout {rollout_index} for '{tensor_key}' should have shape (T, 2), "
                    f"got {x_T_2.shape}"
                )
            if not np.all(np.isfinite(x_T_2)):
                raise ValueError(
                    f"rollout {rollout_index} for '{tensor_key}' contains non-finite values"
                )

        figure, axes = plt.subplots(figsize=(8, 8))
        time_norm = Normalize(vmin=0, vmax=300, clip=True)
        color_map = LinearSegmentedColormap.from_list(
            "LightGreenToDarkBlue",
            ["#b7f7c1", "#4fbf9f", "#1f6fb5", "#08306b"],
        )

        for _, x_T_2 in selected_trajectories:
            smoothed_T_2 = cls._kalman_filter_2d(x_T_2)
            sample_indices = np.arange(0, len(smoothed_T_2), sample_every)
            if sample_indices[-1] != len(smoothed_T_2) - 1:
                sample_indices = np.append(sample_indices, len(smoothed_T_2) - 1)
            sampled_T_2 = smoothed_T_2[sample_indices]

            if len(sampled_T_2) == 1:
                axes.scatter(
                    sampled_T_2[0, 0],
                    sampled_T_2[0, 1],
                    color=color_map(time_norm(0)),
                    s=18,
                )
            else:
                segments = np.stack((sampled_T_2[:-1], sampled_T_2[1:]), axis=1)
                line_collection = LineCollection(
                    segments,
                    cmap=color_map,
                    norm=time_norm,
                    linewidth=2,
                    alpha=0.4,
                )
                line_collection.set_array(sample_indices[:-1])
                axes.add_collection(line_collection)

            axes.scatter(
                smoothed_T_2[0, 0],
                smoothed_T_2[0, 1],
                color="green",
                marker="o",
                s=45,
                zorder=3,
            )
            axes.scatter(
                smoothed_T_2[-1, 0],
                smoothed_T_2[-1, 1],
                color="red",
                marker="x",
                s=55,
                linewidths=2,
                zorder=4,
            )

        axes.scatter([], [], color="green", marker="o", s=45, label="Start")
        axes.scatter([], [], color="red", marker="x", s=55, linewidths=2, label="End")

        axes.set_xlim(-1.1, 1.1)
        axes.set_ylim(-1.1, 1.1)
        axes.set_aspect("equal", adjustable="box")
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(f"2D trajectories: {tensor_key}")
        axes.grid(alpha=0.2)
        axes.legend()

        figure.tight_layout()

        if output_path is not None:
            image_path = Path(output_path)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(image_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()

        return figure, axes

    @staticmethod
    def _kalman_filter_2d(x_T_2: np.ndarray) -> np.ndarray:
        """Smooth a 2D trajectory with a constant-velocity Kalman filter."""
        if len(x_T_2) <= 1:
            return x_T_2.copy()

        transition = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        measurement = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        process_covariance = np.eye(4) * 0.01
        measurement_covariance = np.eye(2)
        state_covariance = np.eye(4)
        state = np.array(
            [
                x_T_2[0, 0],
                x_T_2[0, 1],
                x_T_2[1, 0] - x_T_2[0, 0],
                x_T_2[1, 1] - x_T_2[0, 1],
            ],
            dtype=np.float64,
        )
        identity = np.eye(4)
        smoothed_T_2 = np.empty_like(x_T_2, dtype=np.float64)
        smoothed_T_2[0] = state[:2]

        for time_index in range(1, len(x_T_2)):
            predicted_state = transition @ state
            predicted_covariance = (
                transition @ state_covariance @ transition.T + process_covariance
            )
            innovation = x_T_2[time_index] - measurement @ predicted_state
            innovation_covariance = (
                measurement @ predicted_covariance @ measurement.T
                + measurement_covariance
            )
            kalman_gain = (
                predicted_covariance
                @ measurement.T
                @ np.linalg.inv(innovation_covariance)
            )
            state = predicted_state + kalman_gain @ innovation
            state_covariance = (
                identity - kalman_gain @ measurement
            ) @ predicted_covariance
            smoothed_T_2[time_index] = state[:2]

        return smoothed_T_2

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("tensor keys should be non-empty strings")
        if "/" in key:
            raise ValueError(f"tensor key should not contain '/': {key}")

    @staticmethod
    def _to_numpy(tensor: TemporalTensor, key: str) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            detached = tensor.detach().cpu()
            if detached.dtype == torch.bfloat16:
                detached = detached.float()
            array = detached.numpy()
        elif isinstance(tensor, np.ndarray):
            array = tensor
        else:
            raise TypeError(
                f"tensor_dict['{key}'] should be torch.Tensor or np.ndarray, got {type(tensor)}"
            )

        if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
            raise TypeError(f"tensor_dict['{key}'] has unsupported dtype {array.dtype}")
        return np.ascontiguousarray(array)

    def _initialize_buffers(
            self,
            arrays: Mapping[str, np.ndarray],
            batch_size: int,
            chunk_time: int,
            ) -> None:
        self._batch_size = batch_size
        self._time_capacity = max(self.initial_time_capacity, chunk_time)
        self._buffers = {
            key: np.empty(
                (batch_size, self._time_capacity, x_B_T_D.shape[2]),
                dtype=x_B_T_D.dtype,
            )
            for key, x_B_T_D in arrays.items()
        }

    def _validate_existing_schema(
            self,
            arrays: Mapping[str, np.ndarray],
            batch_size: int,
            ) -> None:
        if set(arrays) != set(self._buffers):
            raise ValueError(
                f"tensor keys changed within one rollout: expected {sorted(self._buffers)}, "
                f"got {sorted(arrays)}"
            )
        if batch_size != self._batch_size:
            raise ValueError(
                f"batch size changed within one rollout: expected {self._batch_size}, got {batch_size}"
            )

        for key, x_B_T_D in arrays.items():
            buffer_B_T_D = self._buffers[key]
            if x_B_T_D.shape[2] != buffer_B_T_D.shape[2]:
                raise ValueError(
                    f"feature dimension changed for '{key}': expected {buffer_B_T_D.shape[2]}, "
                    f"got {x_B_T_D.shape[2]}"
                )
            if x_B_T_D.dtype != buffer_B_T_D.dtype:
                raise ValueError(
                    f"dtype changed for '{key}': expected {buffer_B_T_D.dtype}, got {x_B_T_D.dtype}"
                )

    def _ensure_time_capacity(self, required_time: int) -> None:
        if required_time <= self._time_capacity:
            return

        new_capacity = self._time_capacity
        while new_capacity < required_time:
            new_capacity *= 2

        expanded_buffers: Dict[str, np.ndarray] = {}
        for key, buffer_B_T_D in self._buffers.items():
            expanded_B_T_D = np.empty(
                (buffer_B_T_D.shape[0], new_capacity, buffer_B_T_D.shape[2]),
                dtype=buffer_B_T_D.dtype,
            )
            expanded_B_T_D[:, :self._time_length, :] = buffer_B_T_D[:, :self._time_length, :]
            expanded_buffers[key] = expanded_B_T_D

        self._buffers = expanded_buffers
        self._time_capacity = new_capacity

    def _initialize_or_validate_file(self, h5_file: h5py.File) -> None:
        if "format" not in h5_file.attrs:
            if len(h5_file) > 0 or len(h5_file.attrs) > 0:
                raise ValueError("existing HDF5 file does not use the TensorVisHelper format")
            self._initialize_file(h5_file)
            return

        self._validate_file_header(h5_file)
        schema_group = h5_file["schema"]
        if set(schema_group.keys()) != set(self._buffers.keys()):
            raise ValueError(
                f"tensor keys do not match the file schema: expected {sorted(schema_group.keys())}, "
                f"got {sorted(self._buffers.keys())}"
            )

        for key, buffer_B_T_D in self._buffers.items():
            expected_feature_dim = int(schema_group[key].attrs["feature_dim"])
            expected_dtype = np.dtype(self._decode_attr(schema_group[key].attrs["dtype"]))
            if buffer_B_T_D.shape[2] != expected_feature_dim:
                raise ValueError(
                    f"feature dimension for '{key}' does not match the file schema: "
                    f"expected {expected_feature_dim}, got {buffer_B_T_D.shape[2]}"
                )
            if buffer_B_T_D.dtype != expected_dtype:
                raise ValueError(
                    f"dtype for '{key}' does not match the file schema: "
                    f"expected {expected_dtype}, got {buffer_B_T_D.dtype}"
                )

    def _initialize_file(self, h5_file: h5py.File) -> None:
        h5_file.attrs["format"] = self.FILE_FORMAT
        h5_file.attrs["format_version"] = self.FORMAT_VERSION
        h5_file.attrs["next_rollout_index"] = 0
        schema_group = h5_file.create_group("schema")
        h5_file.create_group("rollouts")

        for key, buffer_B_T_D in self._buffers.items():
            tensor_schema = schema_group.create_group(key)
            tensor_schema.attrs["feature_dim"] = buffer_B_T_D.shape[2]
            tensor_schema.attrs["dtype"] = buffer_B_T_D.dtype.str

        schema_group.attrs["tensor_keys"] = json.dumps(sorted(self._buffers.keys()))

    @classmethod
    def _validate_file_header(cls, h5_file: h5py.File) -> None:
        file_format = cls._decode_attr(h5_file.attrs.get("format", ""))
        if file_format != cls.FILE_FORMAT:
            raise ValueError(f"unsupported tensor visualization file format: {file_format!r}")

        format_version = int(h5_file.attrs.get("format_version", -1))
        if format_version != cls.FORMAT_VERSION:
            raise ValueError(
                f"unsupported tensor visualization format version: {format_version}"
            )
        if "schema" not in h5_file or "rollouts" not in h5_file:
            raise ValueError("tensor visualization file is missing schema or rollouts group")
        if "next_rollout_index" not in h5_file.attrs:
            raise ValueError("tensor visualization file is missing next_rollout_index")

    @staticmethod
    def _decode_attr(value):
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def _reset_buffers(self) -> None:
        self._buffers = {}
        self._batch_size = None
        self._time_length = 0
        self._time_capacity = 0
