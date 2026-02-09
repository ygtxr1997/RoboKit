import os
import yaml
import h5py
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

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

    - Build a global frame index: global_i -> (demo_key, t)
    - __getitem__(i) returns ALL content for that frame as nested dict:
        {
          "demo_key": str,
          "t": int,
          "actions": (7,),
          "dones": scalar,
          "rewards": scalar,
          "robot_states": (...,) (if exists)
          "states": (...,) (if exists)
          "obs": { all obs datasets at time t }
          "meta": { optional env_args/global attrs if you want }
        }

    Assumes HDF5 structure: /data/demo_x/...
    """

    def __init__(self, h5_path: str, load_to_memory: bool = False):
        self.h5_path = h5_path
        self.load_to_memory = load_to_memory

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
            self._h5 = h5py.File(self.h5_path, "r")
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

            self.demo_lengths = []
            for dk in self.demo_keys:
                g = f["data"][dk]
                # prefer obs length
                if "obs" in g:
                    # pick the first obs dataset
                    obs_keys = list(g["obs"].keys())
                    assert len(obs_keys) > 0, f"{dk} has empty obs"
                    T = int(g["obs"][obs_keys[0]].shape[0])
                else:
                    # fallback to actions
                    T = int(g["actions"].shape[0])
                self.demo_lengths.append(T)

        self.prefix = [0] * len(self.demo_lengths)
        for i in range(1, len(self.prefix)):
            self.prefix[i] = self.prefix[i - 1] + self.demo_lengths[i - 1]
        self.total_len = int(sum(self.demo_lengths))

    def _preload_all(self):
        """
        Preload everything into RAM. Warning: can be huge due to images.
        A safer approach is to keep load_to_memory=False.
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

        # binary search over prefix
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

        # should never happen
        dk = self.demo_keys[-1]
        return dk, int(self.demo_lengths[-1] - 1)

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
            "demo_key": dk, "t": t,
            "obs": {},
            "text_instruction": self.text_instruction,
        }

        # datasets at demo level
        for k in ["actions", "dones", "rewards", "robot_states", "states", "replay_reset_every"]:
            if k in g and isinstance(g[k], h5py.Dataset):
                ds = g[k]
                # replay_reset_every is shape (1,) so return scalar/array
                out[k] = ds[()] if ds.shape == () or ds.shape == (1,) else ds[t]

        # obs group
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

    Interval convention (UNIFIED):
      - All index ranges are half-open intervals: [start, end)
      - Python range(start, end) also represents [start, end)

    __getitem__(i) windows (in LOCAL index li inside its dataset):
      - history obs/state indices: [li - n_obs + 1, li + 1)          (length = n_obs)
          => frames: li-n_obs+1, ..., li
      - future  obs/state indices: [li + 1, li + 1 + chunk_size)     (length = chunk_size)
          => frames: li+1, ..., li+chunk_size
      - action  indices:          [li, li + chunk_size)              (length = chunk_size)
          => frames: li, ..., li+chunk_size-1

    Cross-demo behavior:
      - drop_cross_demo=True:
          If any requested index lies outside the current demo range, raise IndexError.
      - drop_cross_demo=False (DEFAULT):
          If requested index is outside the current demo range, use edge-repeat:
            idx < demo_start -> demo_start
            idx >= demo_end  -> demo_end-1
    """

    def __init__(
        self,
        datasets: List[LiberoH5FrameDataset],
        n_obs: int,
        chunk_size: int,
        # which keys are considered "obs/state" for windows
        obs_keys: Optional[List[str]] = None,          # if None => use all obs keys from first sample
        state_keys: Optional[List[str]] = None,        # e.g. ["states", "robot_states"]
        action_key: str = "actions",
        # padding behavior
        drop_cross_demo: bool = False,
        pad_value_numeric: float = 0.0,
        pad_value_image: int = 0,
        max_len: Optional[int] = None,
        # others
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

        # build merged prefix
        self.lengths = [len(ds) for ds in datasets]
        self.prefix = [0] * len(self.lengths)
        for i in range(1, len(self.prefix)):
            self.prefix[i] = self.prefix[i - 1] + self.lengths[i - 1]
        self.total_len = int(sum(self.lengths))
        if self.max_len is not None:
            self.total_len = min(self.total_len, self.max_len)

        # infer keys
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

        # record shapes/dtypes for padding
        self._spec = self._infer_padding_spec(sample)

        # cache demo bounds per (dataset_id, demo_key): (demo_start, demo_end) as [start, end)
        self._demo_bounds_cache: Dict[Tuple[int, str], Tuple[int, int]] = {}

    def __len__(self) -> int:
        return self.total_len

    def _infer_padding_spec(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a spec for padding based on first sample.
        """
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
        else:
            return np.full(shape, self.pad_value_numeric, dtype=dtype)

    def _fetch_frame(self, dataset_id: int, local_i: int) -> Dict[str, Any]:
        return self.datasets[dataset_id][local_i]

    def _safe_fetch_same_demo(self, dataset_id: int, local_i: int, ref_demo_key: str) -> Optional[Dict[str, Any]]:
        """
        Safe fetch for the same dataset and same demo.

        Returns:
          - frame dict if (0 <= local_i < len(ds)) AND demo_key matches
          - None otherwise
        """
        if local_i < 0 or local_i >= self.lengths[dataset_id]:
            return None
        fr = self._fetch_frame(dataset_id, local_i)
        if fr["demo_key"] != ref_demo_key:
            return None
        return fr

    def _get_demo_bounds(self, dataset_id: int, ref_local_i: int, demo_key: str) -> Tuple[int, int]:
        """
        Get local demo bounds as a half-open interval [demo_start, demo_end).

        Meaning:
          - demo_start is the first local index in this dataset whose demo_key == demo_key
          - demo_end is the first local index AFTER the demo (exclusive)
          - valid local indices within the demo are: demo_start, ..., demo_end-1

        Cached per (dataset_id, demo_key).
        """
        cache_key = (int(dataset_id), str(demo_key))
        if cache_key in self._demo_bounds_cache:
            return self._demo_bounds_cache[cache_key]

        L = self.lengths[dataset_id]
        # Scan left to find demo_start
        demo_start = ref_local_i
        while demo_start - 1 >= 0:
            fr = self._fetch_frame(dataset_id, demo_start - 1)
            if fr["demo_key"] != demo_key:
                break
            demo_start -= 1

        # Scan right to find demo_end (exclusive)
        demo_end = ref_local_i + 1
        while demo_end < L:
            fr = self._fetch_frame(dataset_id, demo_end)
            if fr["demo_key"] != demo_key:
                break
            demo_end += 1

        self._demo_bounds_cache[cache_key] = (demo_start, demo_end)
        return demo_start, demo_end

    def _edge_repeat_fetch(self, dataset_id: int, idx: int, ref_local_i: int, demo_key: str) -> Dict[str, Any]:
        """
        Fetch frame using edge-repeat within the current demo bounds [demo_start, demo_end).

        - If idx < demo_start: clamp to demo_start
        - If idx >= demo_end:  clamp to demo_end-1
        """
        demo_start, demo_end = self._get_demo_bounds(dataset_id, ref_local_i, demo_key)
        # clamp into [demo_start, demo_end-1]
        if idx < demo_start:
            idx = demo_start
        elif idx >= demo_end:
            idx = demo_end - 1

        fr = self._fetch_frame(dataset_id, idx)
        # Sanity check: should always match
        if fr["demo_key"] != demo_key:
            # Fallback (shouldn't happen if bounds are correct)
            fr = self._fetch_frame(dataset_id, demo_start)
        return fr

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ds_id, li = self._locate_dataset(i)
        cur = self._fetch_frame(ds_id, li)
        demo_key = cur["demo_key"]

        # Build index ranges as HALF-OPEN intervals [start, end)
        # history: [li - n_obs + 1, li + 1)
        hist_start = li - self.n_obs + 1
        hist_end   = li + 1

        # future:  [li + 1, li + 1 + chunk_size)
        fut_start  = li + 1
        fut_end    = li + 1 + self.chunk_size

        # action:  [li, li + chunk_size)
        act_start  = li
        act_end    = li + self.chunk_size

        hist_indices = list(range(hist_start, hist_end))  # length n_obs
        fut_indices  = list(range(fut_start,  fut_end))   # length chunk_size
        act_indices  = list(range(act_start,  act_end))   # length chunk_size

        # demo bounds in the same half-open convention [demo_start, demo_end)
        demo_start, demo_end = self._get_demo_bounds(ds_id, li, demo_key)

        if self.drop_cross_demo:
            # strict: any idx outside [demo_start, demo_end) is illegal
            for idx in hist_indices + fut_indices + act_indices:
                if not (demo_start <= idx < demo_end):
                    raise IndexError(
                        f"Window crosses demo boundary at global i={i}, local idx={idx}, "
                        f"demo_range=[{demo_start},{demo_end})"
                    )

            def get_frame(idx: int) -> Dict[str, Any]:
                # safe because checked above
                return self._fetch_frame(ds_id, idx)
        else:
            # default: edge-repeat within demo
            def get_frame(idx: int) -> Dict[str, Any]:
                return self._edge_repeat_fetch(ds_id, idx, li, demo_key)

        # assemble history
        hist_obs = {k: [] for k in self.obs_keys}
        hist_state = {k: [] for k in self.state_keys}
        for idx in hist_indices:
            fr = get_frame(idx)
            for ok in self.obs_keys:
                hist_obs[ok].append(fr["obs"][ok])
            for sk in self.state_keys:
                hist_state[sk].append(fr[sk])

        # assemble future
        if self.load_future_obs:
            fut_obs = {k: [] for k in self.obs_keys}
            fut_state = {k: [] for k in self.state_keys}
            for idx in fut_indices:
                fr = get_frame(idx)
                for ok in self.obs_keys:
                    fut_obs[ok].append(fr["obs"][ok])
                for sk in self.state_keys:
                    fut_state[sk].append(fr[sk])

        # assemble actions
        action_list = []
        for idx in act_indices:
            fr = get_frame(idx)
            action_list.append(fr[self.action_key])

        # stack
        ret_sample = {
            "demo_key": demo_key,
            "t": int(cur["t"]),
            "text_instruction": cur["text_instruction"],
        }
        hist_obs = {k: np.stack(v, axis=0) for k, v in hist_obs.items()}
        hist_state = {k: np.stack(v, axis=0) for k, v in hist_state.items()}
        ret_sample["history"] = {"obs": hist_obs, "state": hist_state}
        if self.load_future_obs:
            fut_obs = {k: np.stack(v, axis=0) for k, v in fut_obs.items()}
            fut_state = {k: np.stack(v, axis=0) for k, v in fut_state.items()}
            ret_sample["future"] = {"obs": fut_obs, "state": fut_state}
        action_arr = np.stack(action_list, axis=0)
        ret_sample["action"] = action_arr

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

        Returns:
          {
            "obs.wrenches": (N, dim),
            "state.states": (N, dim),
            "action.actions": (N, dim),
            ...
          }
        """
        collected: Dict[str, List[np.ndarray]] = {}

        # init lists
        for k in keys["obs"]:
            collected[f"obs.{k}"] = []
        for k in keys["state"]:
            collected[f"state.{k}"] = []
        # action: keep one key
        collected[f"action.{self.action_key}"] = []

        # iterate each underlying dataset, read demo-by-demo in contiguous slices
        for ds in self.datasets:
            f = ds._get_h5()
            for dk in ds.demo_keys:
                g = f["data"][dk]

                # length T
                if "obs" in g:
                    obs_keys = list(g["obs"].keys())
                    if len(obs_keys) > 0:
                        T = int(g["obs"][obs_keys[0]].shape[0])
                    else:
                        T = int(g["actions"].shape[0])
                else:
                    T = int(g["actions"].shape[0])

                # ---- obs (numeric only; keys already filtered outside) ----
                if "obs" in g:
                    for ok in keys["obs"]:
                        if ok not in g["obs"]:
                            continue
                        arr = g["obs"][ok][...]              # (T, ...)
                        collected[f"obs.{ok}"].append(arr.reshape(T, -1))

                # ---- state (demo-level datasets like "states"/"robot_states") ----
                for sk in keys["state"]:
                    if sk in g and isinstance(g[sk], h5py.Dataset):
                        arr = g[sk][...]                    # (T, ...)
                        collected[f"state.{sk}"].append(arr.reshape(T, -1))

                # ---- action (demo-level) ----
                if self.action_key in g and isinstance(g[self.action_key], h5py.Dataset):
                    arr = g[self.action_key][...]           # (T, action_dim) or (T, ...)
                    collected[f"action.{self.action_key}"].append(arr.reshape(T, -1))

        # concat across all demos/datasets
        out: Dict[str, np.ndarray] = {}
        for name, parts in collected.items():
            if len(parts) == 0:
                continue
            out[name] = np.concatenate(parts, axis=0)
        return out

    def compute_statistics(
        self,
        keys: Optional[Dict[str, List[str]]] = None,
        quantile_max_samples: int = 300000,  # kept for API compatibility; used as optional downsample cap
        seed: int = 0,
        verbose_every: int = 50000,          # unused in this basic all-array mode
    ) -> Dict[str, Any]:
        """
        Basic implementation:
          1) Read ALL numeric arrays (no windows, no images) into memory
          2) Compute min/max/mean/std and p01/p99 (optionally subsample for quantile)
        """
        # default keys: numeric obs (skip uint8) + all state_keys + action_key
        if keys is None:
            keys = {"obs": [], "state": [], "action": [self.action_key]}
            for ok in self.obs_keys:
                dt = self._spec["obs"][ok]["dtype"]
                if not np.issubdtype(dt, np.uint8):
                    keys["obs"].append(ok)
            keys["state"] = list(self.state_keys)

        rng = np.random.RandomState(seed)

        # 1) collect big arrays
        arrays = self._collect_current_arrays(keys)

        # 2) compute stats
        out: Dict[str, Any] = {}

        for name, X in arrays.items():
            # X: (N, dim)
            X64 = X.astype(np.float64, copy=False)
            count = int(X64.shape[0])

            mean = np.mean(X64, axis=0)
            std = np.std(X64, axis=0, ddof=1) if count > 1 else np.zeros_like(mean)
            minv = np.min(X64, axis=0)
            maxv = np.max(X64, axis=0)

            # quantiles: basic approach = use all, but allow cap via random subset
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


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # 1) single dataset
    ds1 = LiberoH5FrameDataset("dataset1.hdf5", load_to_memory=False)
    print("len(ds1) =", len(ds1))
    x = ds1[123]
    print(x["demo_key"], x["t"], list(x["obs"].keys()))

    # 2) merged dataset with windows
    ds2 = LiberoH5FrameDataset("dataset2.hdf5", load_to_memory=False)
    merged = MergeDataset(
        datasets=[ds1, ds2],
        n_obs=4,
        chunk_size=8,
        obs_keys=None,                 # auto: all obs keys
        state_keys=["states"],         # e.g. ["states","robot_states"] if you want
        action_key="actions",
        drop_cross_demo=False,
    )
    print("len(merged) =", len(merged))
    item = merged[1000]
    print(item["history"]["obs"].keys(), item["action"].shape)

    # 3) compute stats and save yaml
    stats = merged.compute_statistics(
        keys=None,                     # auto numeric obs + states + actions
        quantile_max_samples=200000,
        seed=0,
        verbose_every=20000,
    )
    MergeDataset.save_statistics_yaml(stats, "out/statistics.yaml")

    ds1.close()
    ds2.close()
