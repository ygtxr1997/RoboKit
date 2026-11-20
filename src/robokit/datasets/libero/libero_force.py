import os
import yaml
import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import bisect

from robokit.debug_utils.printer import beautiful_print


# =========================
# Utils: streaming stats
# =========================

@dataclass
class StreamStats:
    """Online mean/std + min/max for vector data."""
    dim: int
    n: int = 0
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None
    minv: Optional[np.ndarray] = None
    maxv: Optional[np.ndarray] = None

    def __post_init__(self):
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.M2 = np.zeros((self.dim,), dtype=np.float64)
        self.minv = np.full((self.dim,), np.inf, dtype=np.float64)
        self.maxv = np.full((self.dim,), -np.inf, dtype=np.float64)

    def update_batch(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        assert x.ndim == 2 and x.shape[1] == self.dim, (x.shape, self.dim)

        self.minv = np.minimum(self.minv, np.min(x, axis=0))
        self.maxv = np.maximum(self.maxv, np.max(x, axis=0))

        for i in range(x.shape[0]):
            self.n += 1
            delta = x[i] - self.mean
            self.mean += delta / self.n
            delta2 = x[i] - self.mean
            self.M2 += delta * delta2

    def finalize(self) -> Dict[str, Any]:
        var = self.M2 / max(self.n - 1, 1)
        std = np.sqrt(np.maximum(var, 1e-12))
        return {
            "count": int(self.n),
            "mean": self.mean,
            "std": std,
            "min": self.minv,
            "max": self.maxv,
        }


def _to_py(v):
    """Convert numpy objects to YAML-friendly python types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _quantiles_from_samples(samples: List[np.ndarray], q01=0.01, q99=0.99) -> Tuple[np.ndarray, np.ndarray]:
    if len(samples) == 0:
        raise ValueError("No samples provided for quantiles")
    x = np.concatenate(samples, axis=0).astype(np.float64)
    p01 = np.quantile(x, q01, axis=0)
    p99 = np.quantile(x, q99, axis=0)
    return p01, p99


# =========================
# 1) Single HDF5 dataset
# =========================

class LiberoH5FrameDataset:
    """
    Minimal per-frame dataset.

    Assumes HDF5 structure: /data/demo_x/...
    """

    def __init__(
        self,
        h5_path: str,
        load_to_memory: bool = False,
        rdcc_nbytes: int = 128 * 1024 * 1024,
        rdcc_nslots: int = 1_000_003,
        rdcc_w0: float = 0.75,
        demo_keys: Optional[List[str]] = None,
    ):
        self.h5_path = h5_path
        self.selected_demo_keys = demo_keys
        self.load_to_memory = load_to_memory

        # HDF5 raw data chunk cache params (speed-up for repeated chunk reads)
        self.rdcc_nbytes = int(rdcc_nbytes)
        self.rdcc_nslots = int(rdcc_nslots)
        self.rdcc_w0 = float(rdcc_w0)

        self._h5 = None  # lazy open (safe for multiprocessing)
        self.text_instruction = ""
        self.demo_keys: List[str] = []
        self.demo_lengths: List[int] = []
        self.prefix: List[int] = []
        self.total_len: int = 0

        # optional cache if load_to_memory
        self._cache: Dict[str, Any] = {}

        # build mapping
        self._scan()

        # if load_to_memory, preload (optional but can be huge for images)
        if self.load_to_memory:
            self._preload_all()

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=self.rdcc_nbytes,
                rdcc_nslots=self.rdcc_nslots,
                rdcc_w0=self.rdcc_w0,
            )
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    def _scan(self):
        with h5py.File(self.h5_path, "r") as f:
            assert "data" in f, "HDF5 must contain top-level group 'data'"
            bddl_file_name = f["data"].attrs["bddl_file_name"]
            self.text_instruction = os.path.basename(bddl_file_name).split(".")[0].replace("_", " ")
            self.demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]))

            if self.selected_demo_keys is not None:
                missing = set(self.selected_demo_keys) - set(self.demo_keys)
                if missing:
                    raise ValueError(f"Unknown demo keys in {self.h5_path}: {sorted(missing)}")
                # Filtering here also limits MergeDataset's statistics to this split.
                selected = set(self.selected_demo_keys)
                self.demo_keys = [key for key in self.demo_keys if key in selected]

            self.demo_lengths = []
            for dk in self.demo_keys:
                g = f["data"][dk]
                if "obs" in g:
                    obs_keys = list(g["obs"].keys())
                    assert len(obs_keys) > 0, f"{dk} has empty obs"
                    T = int(g["obs"][obs_keys[0]].shape[0])
                else:
                    T = int(g["actions"].shape[0])
                self.demo_lengths.append(T)

        self.prefix = [0] * len(self.demo_lengths)
        for i in range(1, len(self.prefix)):
            self.prefix[i] = self.prefix[i - 1] + self.demo_lengths[i - 1]
        self.total_len = int(sum(self.demo_lengths))

    def _preload_all(self):
        """
        Preload everything into RAM. Warning: can be huge due to images.
        """
        f = self._get_h5()
        self._cache["data"] = {}
        for dk in self.demo_keys:
            g = f["data"][dk]
            demo_dict = {}
            for k in g.keys():
                if isinstance(g[k], h5py.Dataset):
                    demo_dict[k] = g[k][...]
                elif isinstance(g[k], h5py.Group) and k == "obs":
                    demo_dict["obs"] = {ok: g["obs"][ok][...] for ok in g["obs"].keys()}
            self._cache["data"][dk] = demo_dict

    def __len__(self) -> int:
        return self.total_len

    def _locate(self, i: int) -> Tuple[str, int]:
        """global frame i -> (demo_key, t_in_demo)"""
        i = int(i)
        if i < 0 or i >= self.total_len:
            raise IndexError(i)

        lo, hi = 0, len(self.prefix) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.prefix[mid]
            T = self.demo_lengths[mid]
            if i < start:
                hi = mid - 1
            elif i >= start + T:
                lo = mid + 1
            else:
                dk = self.demo_keys[mid]
                return dk, int(i - start)

        dk = self.demo_keys[-1]
        return dk, int(self.demo_lengths[-1] - 1)

    def _locate_demo_idx_by_local(self, local_i: int) -> int:
        local_i = int(local_i)
        if local_i < 0 or local_i >= self.total_len:
            raise IndexError(local_i)
        idx = bisect.bisect_right(self.prefix, local_i) - 1
        if idx < 0:
            idx = 0
        return idx

    def get_demo_window_info(self, local_i: int) -> Tuple[str, int, int, int]:
        """
        Return (demo_key, demo_start, demo_end, t_in_demo), all in dataset-local indexing.
        demo interval uses half-open [demo_start, demo_end).
        """
        demo_idx = self._locate_demo_idx_by_local(local_i)
        demo_start = self.prefix[demo_idx]
        demo_end = demo_start + self.demo_lengths[demo_idx]
        demo_key = self.demo_keys[demo_idx]
        t_in_demo = int(local_i - demo_start)
        return demo_key, int(demo_start), int(demo_end), t_in_demo

    def get_demo_data(self, demo_key: str):
        """
        Return demo container:
          - h5py.Group if load_to_memory=False
          - dict if load_to_memory=True
        """
        if self.load_to_memory:
            return self._cache["data"][demo_key]
        f = self._get_h5()
        return f["data"][demo_key]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        dk, t = self._locate(i)

        if self.load_to_memory:
            g = self._cache["data"][dk]
            out = {
                "demo_key": dk,
                "t": t,
                "obs": {ok: g["obs"][ok][t] for ok in g.get("obs", {}).keys()},
            }
            for k in ["actions", "dones", "rewards", "robot_states", "states"]:
                if k in g:
                    out[k] = g[k][t]
            return out

        f = self._get_h5()
        g = f["data"][dk]

        out: Dict[str, Any] = {
            "demo_key": dk,
            "t": t,
            "obs": {},
            "text_instruction": self.text_instruction,
        }

        for k in ["actions", "dones", "rewards", "robot_states", "states", "replay_reset_every"]:
            if k in g and isinstance(g[k], h5py.Dataset):
                ds = g[k]
                out[k] = ds[()] if ds.shape == () or ds.shape == (1,) else ds[t]

        if "obs" in g:
            for ok in g["obs"].keys():
                out["obs"][ok] = g["obs"][ok][t]

        return out


# =========================
# 2) MergeDataset with windows + stats
# =========================

class MergeDataset:
    """
    Merge multiple LiberoH5FrameDataset into one global frame index.
    """

    def __init__(
        self,
        datasets: List[LiberoH5FrameDataset],
        n_obs: int,
        chunk_size: int,
        obs_keys: Optional[List[str]] = None,
        state_keys: Optional[List[str]] = None,
        action_key: str = "actions",
        drop_cross_demo: bool = False,
        pad_value_numeric: float = 0.0,
        pad_value_image: int = 0,
        max_len: Optional[int] = None,
        load_future_obs: bool = False,
    ):
        assert len(datasets) > 0
        self.datasets = datasets
        self.n_obs = int(n_obs)
        self.chunk_size = int(chunk_size)
        self.action_key = action_key
        self.drop_cross_demo = drop_cross_demo
        self.pad_value_numeric = float(pad_value_numeric)
        self.pad_value_image = int(pad_value_image)
        self.max_len = None if max_len is None else int(max_len)
        self.load_future_obs = load_future_obs

        self.lengths = [len(ds) for ds in datasets]
        self.prefix = [0] * len(self.lengths)
        for i in range(1, len(self.prefix)):
            self.prefix[i] = self.prefix[i - 1] + self.lengths[i - 1]
        self.total_len = int(sum(self.lengths))
        if self.max_len is not None:
            self.total_len = min(self.total_len, self.max_len)

        sample = self.datasets[0][0]
        if obs_keys is None:
            obs_keys = list(sample.get("obs", {}).keys())
        if state_keys is None:
            state_keys = []
            for k in ["states", "robot_states"]:
                if k in sample:
                    state_keys.append(k)
        self.obs_keys = obs_keys
        self.state_keys = state_keys

        self._spec = self._infer_padding_spec(sample)

    def __len__(self) -> int:
        return self.total_len

    def _infer_padding_spec(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        spec = {"obs": {}, "state": {}, "action": None}
        for ok in self.obs_keys:
            arr = sample["obs"][ok]
            spec["obs"][ok] = {"shape": arr.shape, "dtype": arr.dtype}
        for sk in self.state_keys:
            arr = sample[sk]
            spec["state"][sk] = {"shape": arr.shape, "dtype": arr.dtype}
        if self.action_key in sample:
            arr = sample[self.action_key]
            spec["action"] = {"shape": arr.shape, "dtype": arr.dtype}
        return spec

    def _locate_dataset(self, i: int) -> Tuple[int, int]:
        """global i -> (dataset_id, local_i)"""
        i = int(i)
        if i < 0 or i >= self.total_len:
            raise IndexError(i)
        lo, hi = 0, len(self.prefix) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.prefix[mid]
            L = self.lengths[mid]
            if i < start:
                hi = mid - 1
            elif i >= start + L:
                lo = mid + 1
            else:
                return mid, int(i - start)
        return len(self.prefix) - 1, int(self.lengths[-1] - 1)

    def _make_pad_like(self, shape, dtype):
        if np.issubdtype(dtype, np.uint8):
            return np.full(shape, self.pad_value_image, dtype=dtype)
        return np.full(shape, self.pad_value_numeric, dtype=dtype)

    def _fetch_frame(self, dataset_id: int, local_i: int) -> Dict[str, Any]:
        return self.datasets[dataset_id][local_i]

    @staticmethod
    def _take_rows_h5safe(arr, idx: np.ndarray):
        """
        h5py advanced indexing requires strictly increasing indices.
        This helper supports duplicated / non-strict indices by
        reading unique sorted indices once and gathering back.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"idx must be 1D, got shape={idx.shape}")
        if idx.size == 0:
            return arr[idx]
        uniq, inv = np.unique(idx, return_inverse=True)  # uniq sorted asc
        gathered = arr[uniq]
        return gathered[inv]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ds_id, li = self._locate_dataset(i)
        ds = self.datasets[ds_id]

        # Fast demo lookup without per-frame HDF5 probing.
        demo_key, demo_start, demo_end, t_in_demo = ds.get_demo_window_info(li)

        hist_idx = np.arange(li - self.n_obs + 1, li + 1, dtype=np.int64)  # length n_obs
        fut_idx = np.arange(li + 1, li + 1 + self.chunk_size, dtype=np.int64)  # length chunk_size
        act_idx = np.arange(li, li + self.chunk_size, dtype=np.int64)  # length chunk_size

        if self.drop_cross_demo:
            merged = np.concatenate([hist_idx, fut_idx, act_idx], axis=0)
            if np.any(merged < demo_start) or np.any(merged >= demo_end):
                bad = merged[(merged < demo_start) | (merged >= demo_end)][0]
                raise IndexError(
                    f"Window crosses demo boundary at global i={i}, local idx={int(bad)}, "
                    f"demo_range=[{demo_start},{demo_end})"
                )
        else:
            hist_idx = np.clip(hist_idx, demo_start, demo_end - 1)
            fut_idx = np.clip(fut_idx, demo_start, demo_end - 1)
            act_idx = np.clip(act_idx, demo_start, demo_end - 1)

        # Convert dataset-local idx -> demo-local t for one-shot fancy indexing.
        hist_t = hist_idx - demo_start
        fut_t = fut_idx - demo_start
        act_t = act_idx - demo_start

        g = ds.get_demo_data(demo_key)
        is_mem = ds.load_to_memory

        hist_obs = {}
        for ok in self.obs_keys:
            hist_obs[ok] = self._take_rows_h5safe(g["obs"][ok], hist_t)

        hist_state = {}
        for sk in self.state_keys:
            hist_state[sk] = self._take_rows_h5safe(g[sk], hist_t)

        if self.load_future_obs:
            fut_obs = {}
            for ok in self.obs_keys:
                fut_obs[ok] = self._take_rows_h5safe(g["obs"][ok], fut_t)

            fut_state = {}
            for sk in self.state_keys:
                fut_state[sk] = self._take_rows_h5safe(g[sk], fut_t)

        action_arr = self._take_rows_h5safe(g[self.action_key], act_t)

        ret_sample = {
            "demo_key": demo_key,
            "t": int(t_in_demo),
            "text_instruction": ds.text_instruction,
            "history": {"obs": hist_obs, "state": hist_state},
            "action": action_arr,
        }
        if self.load_future_obs:
            ret_sample["future"] = {"obs": fut_obs, "state": fut_state}

        return ret_sample

    # =========================
    # Statistics
    # =========================

    def _collect_current_arrays(
        self,
        keys: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """
        Collect per-frame "current" arrays (no windows, no images) into big numpy arrays.
        """
        collected: Dict[str, List[np.ndarray]] = {}

        for k in keys["obs"]:
            collected[f"obs.{k}"] = []
        for k in keys["state"]:
            collected[f"state.{k}"] = []
        collected[f"action.{self.action_key}"] = []

        for ds in self.datasets:
            f = ds._get_h5()
            for dk in ds.demo_keys:
                g = f["data"][dk]

                if "obs" in g:
                    obs_keys = list(g["obs"].keys())
                    if len(obs_keys) > 0:
                        T = int(g["obs"][obs_keys[0]].shape[0])
                    else:
                        T = int(g["actions"].shape[0])
                else:
                    T = int(g["actions"].shape[0])

                if "obs" in g:
                    for ok in keys["obs"]:
                        if ok not in g["obs"]:
                            continue
                        arr = g["obs"][ok][...]
                        collected[f"obs.{ok}"].append(arr.reshape(T, -1))

                for sk in keys["state"]:
                    if sk in g and isinstance(g[sk], h5py.Dataset):
                        arr = g[sk][...]
                        collected[f"state.{sk}"].append(arr.reshape(T, -1))

                if self.action_key in g and isinstance(g[self.action_key], h5py.Dataset):
                    arr = g[self.action_key][...]
                    collected[f"action.{self.action_key}"].append(arr.reshape(T, -1))

        out: Dict[str, np.ndarray] = {}
        for name, parts in collected.items():
            if len(parts) == 0:
                continue
            out[name] = np.concatenate(parts, axis=0)
        return out

    def compute_statistics(
        self,
        keys: Optional[Dict[str, List[str]]] = None,
        quantile_max_samples: int = 300000,
        seed: int = 0,
        verbose_every: int = 50000,
    ) -> Dict[str, Any]:
        """
        Basic implementation:
          1) Read ALL numeric arrays (no windows, no images) into memory
          2) Compute min/max/mean/std and p01/p99 (optionally subsample for quantile)
        """
        if keys is None:
            keys = {"obs": [], "state": [], "action": [self.action_key]}
            for ok in self.obs_keys:
                dt = self._spec["obs"][ok]["dtype"]
                if not np.issubdtype(dt, np.uint8):
                    keys["obs"].append(ok)
            keys["state"] = list(self.state_keys)

        rng = np.random.RandomState(seed)
        arrays = self._collect_current_arrays(keys)

        out: Dict[str, Any] = {}
        for name, X in arrays.items():
            X64 = X.astype(np.float64, copy=False)
            count = int(X64.shape[0])

            mean = np.mean(X64, axis=0)
            std = np.std(X64, axis=0, ddof=1) if count > 1 else np.zeros_like(mean)
            minv = np.min(X64, axis=0)
            maxv = np.max(X64, axis=0)

            if count > quantile_max_samples:
                idx = rng.choice(count, size=quantile_max_samples, replace=False)
                Xq = X64[idx]
            else:
                Xq = X64

            p01 = np.quantile(Xq, 0.01, axis=0)
            p99 = np.quantile(Xq, 0.99, axis=0)

            out[name] = {
                "count": count,
                "mean": _to_py(mean),
                "std": _to_py(std),
                "min": _to_py(minv),
                "max": _to_py(maxv),
                "p01": _to_py(p01),
                "p99": _to_py(p99),
            }

        beautiful_print(out)
        return out

    @staticmethod
    def save_statistics_yaml(stats: Dict[str, Any], out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            yaml.safe_dump(stats, f, sort_keys=True, allow_unicode=True)


if __name__ == "__main__":
    ds1 = LiberoH5FrameDataset("dataset1.hdf5", load_to_memory=False)
    print("len(ds1) =", len(ds1))
    x = ds1[123]
    print(x["demo_key"], x["t"], list(x["obs"].keys()))

    ds2 = LiberoH5FrameDataset("dataset2.hdf5", load_to_memory=False)
    merged = MergeDataset(
        datasets=[ds1, ds2],
        n_obs=4,
        chunk_size=8,
        obs_keys=None,
        state_keys=["states"],
        action_key="actions",
        drop_cross_demo=False,
    )
    print("len(merged) =", len(merged))
    item = merged[1000]
    print(item["history"]["obs"].keys(), item["action"].shape)

    stats = merged.compute_statistics(
        keys=None,
        quantile_max_samples=200000,
        seed=0,
        verbose_every=20000,
    )
    MergeDataset.save_statistics_yaml(stats, "out/statistics.yaml")

    ds1.close()
    ds2.close()
