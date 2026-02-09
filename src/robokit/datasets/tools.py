import os
import glob
import numpy as np
import h5py
from typing import Optional, Dict, Iterable, Tuple, Union, List


class T5EmbeddingsH5:
    """
    HDF5 layout:
      /unique_embeddings   float16/float32 [N, 512, 1024]
      /text_valid_lengths  int32           [N]
      /unique_texts        vlen utf-8 str  [N]
    attrs:
      n_unique_texts (int)
      embedding_shape (tuple)
      created_from (str, optional)
    """

    def __init__(
        self,
        h5_path: str,
        mode: str = "a",
        embedding_shape: Tuple[int, int] = (512, 1024),
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
        chunks: Optional[Tuple[int, int, int]] = (1, 512, 1024),
    ):
        self.h5_path = h5_path
        self.mode = mode
        self.embedding_shape = embedding_shape
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunks = chunks

        self.f = h5py.File(h5_path, mode)
        self._ensure_layout()

        self._text2id: Optional[Dict[str, int]] = None

    def _ensure_layout(self):
        if "unique_embeddings" in self.f:
            # existing file, trust stored attrs if present
            if "embedding_shape" in self.f.attrs:
                self.embedding_shape = tuple(self.f.attrs["embedding_shape"])
            return

        # Create fresh datasets (extendable along axis 0)
        emb_dtype = np.float16  # 默认；也可以写入时自动按输入 dtype
        maxshape = (None, *self.embedding_shape)

        self.f.create_dataset(
            "unique_embeddings",
            shape=(0, *self.embedding_shape),
            maxshape=maxshape,
            dtype=emb_dtype,
            chunks=self.chunks,
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression == "gzip" else None,
        )

        self.f.create_dataset(
            "text_valid_lengths",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1024,),
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression == "gzip" else None,
        )

        str_dtype = h5py.string_dtype(encoding="utf-8")
        self.f.create_dataset(
            "unique_texts",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype,
            chunks=(1024,),
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression == "gzip" else None,
        )

        self.f.attrs["n_unique_texts"] = 0
        self.f.attrs["embedding_shape"] = np.array(self.embedding_shape, dtype=np.int32)

    @property
    def n_unique(self) -> int:
        return int(self.f.attrs.get("n_unique_texts", self.f["unique_embeddings"].shape[0]))

    def build_text_index(self) -> Dict[str, int]:
        """
        构建 text->id 映射（会占用一定内存，但查找最快）。
        """
        texts = self.f["unique_texts"][:]
        # h5py 可能返回 bytes，需要 decode
        out = {}
        for i, t in enumerate(texts):
            if isinstance(t, bytes):
                t = t.decode("utf-8")
            out[t] = i
        self._text2id = out
        return out

    def get_id(self, text: str) -> Optional[int]:
        if self._text2id is None:
            self.build_text_index()
        return self._text2id.get(text)

    def append_one(self, text: str, embedding: np.ndarray, valid_length: int) -> int:
        """
        追加一个 (text, embedding, valid_length)，返回其 id。
        注意：默认不做去重；如果要去重，请先 build_text_index + get_id。
        """
        embedding = np.asarray(embedding)
        if embedding.shape != self.embedding_shape:
            raise ValueError(f"embedding.shape={embedding.shape} != expected {self.embedding_shape}")

        ds_emb = self.f["unique_embeddings"]
        ds_len = self.f["text_valid_lengths"]
        ds_txt = self.f["unique_texts"]

        new_id = ds_emb.shape[0]
        ds_emb.resize((new_id + 1, *self.embedding_shape))
        ds_len.resize((new_id + 1,))
        ds_txt.resize((new_id + 1,))

        # 如果首次写入 dtype 与默认 float16 不同，这里做一次对齐
        if ds_emb.dtype != embedding.dtype:
            embedding_to_write = embedding.astype(ds_emb.dtype, copy=False)
        else:
            embedding_to_write = embedding

        ds_emb[new_id] = embedding_to_write
        ds_len[new_id] = np.int32(valid_length)
        ds_txt[new_id] = text

        self.f.attrs["n_unique_texts"] = new_id + 1

        if self._text2id is not None:
            self._text2id[text] = new_id

        return new_id

    def get_by_id(self, idx: int) -> Tuple[str, np.ndarray, int]:
        ds_emb = self.f["unique_embeddings"]
        ds_len = self.f["text_valid_lengths"]
        ds_txt = self.f["unique_texts"]

        t = ds_txt[idx]
        if isinstance(t, bytes):
            t = t.decode("utf-8")
        emb = ds_emb[idx]
        vlen = int(ds_len[idx])
        return t, emb, vlen

    def flush(self):
        self.f.flush()

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------------------- Conversion helpers ----------------------

    @staticmethod
    def convert_npz_to_h5(
        npz_path: str,
        h5_path: Optional[str] = None,
        overwrite: bool = False,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
        chunks: Optional[Tuple[int, int, int]] = (1, 512, 1024),
    ) -> str:
        """
        把单个 npz 转成 h5（同字段名）。
        - 支持 npz 中的:
            unique_embeddings: (N, 512, 1024)
            text_valid_lengths: (N,)
            unique_texts: object array/list[str]
            n_unique_texts: (1,) optional
        """
        if h5_path is None:
            base, _ = os.path.splitext(npz_path)
            h5_path = base + ".h5"

        if os.path.exists(h5_path):
            if overwrite:
                os.remove(h5_path)
            else:
                return h5_path

        cached = np.load(npz_path, allow_pickle=True)

        unique_embeddings = cached["unique_embeddings"]
        text_valid_lengths = cached["text_valid_lengths"]
        unique_texts = cached["unique_texts"]

        # unique_texts 可能是 object array，需要转成 python str 列表
        if isinstance(unique_texts, np.ndarray):
            unique_texts_list = unique_texts.tolist()
        else:
            unique_texts_list = list(unique_texts)

        # embeddings shape check
        if unique_embeddings.ndim != 3:
            raise ValueError(f"unique_embeddings.ndim={unique_embeddings.ndim}, expected 3")
        n, t, d = unique_embeddings.shape
        embedding_shape = (t, d)

        with T5EmbeddingsH5(
            h5_path,
            mode="w",
            embedding_shape=embedding_shape,
            compression=compression,
            compression_opts=compression_opts,
            chunks=(chunks[0], embedding_shape[0], embedding_shape[1]) if chunks else None,
        ) as h5:
            # 如果想保持原 dtype（比如 float32），可以把 ds dtype 改成输入 dtype：
            # 这里简单做：直接把 dataset 重建为输入 dtype
            # 但为了不改动 ensure_layout，采用：重建 unique_embeddings dataset
            del h5.f["unique_embeddings"]
            h5.f.create_dataset(
                "unique_embeddings",
                shape=(n, *embedding_shape),
                maxshape=(None, *embedding_shape),
                dtype=unique_embeddings.dtype,
                chunks=(1, *embedding_shape) if chunks else None,
                compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
            )

            # 写入
            h5.f["text_valid_lengths"].resize((n,))
            h5.f["unique_texts"].resize((n,))

            # embeddings 用 slice 写，避免额外复制（npz 解压后仍在内存，但写入不会再叠加一份）
            ds_emb = h5.f["unique_embeddings"]
            for i in range(n):
                ds_emb[i] = unique_embeddings[i]

            h5.f["text_valid_lengths"][:] = text_valid_lengths.astype(np.int32, copy=False)
            h5.f["unique_texts"][:] = np.array(unique_texts_list, dtype=h5py.string_dtype("utf-8"))

            h5.f.attrs["n_unique_texts"] = int(n)
            h5.f.attrs["created_from"] = os.path.basename(npz_path)

        return h5_path

    @staticmethod
    def batch_convert_dir(
        dir_path: str,
        pattern: str = "t5_embeddings_*.npz",
        overwrite: bool = False,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
        chunks: Optional[Tuple[int, int, int]] = (1, 512, 1024),
    ) -> Dict[str, str]:
        """
        批量把目录下 npz 转成 h5，返回 {npz_path: h5_path}.
        """
        out = {}
        for npz_path in sorted(glob.glob(os.path.join(dir_path, pattern))):
            h5_path = T5EmbeddingsH5.convert_npz_to_h5(
                npz_path=npz_path,
                h5_path=None,
                overwrite=overwrite,
                compression=compression,
                compression_opts=compression_opts,
                chunks=chunks,
            )
            out[npz_path] = h5_path
        return out


class MultiH5Embeddings:
    """
    Manage multiple HDF5 files under a directory.
    Build a global index: text -> (file_idx, local_id)
    Read embeddings lazily (per-process file handles).

    Each H5 file is expected to have:
      - unique_texts [N] (vlen utf-8 string)
      - unique_embeddings [N, 512, 1024]
      - text_valid_lengths [N] (optional)
    """

    def __init__(self, h5_dir: str, pattern: str = "*.h5"):
        self.h5_dir = h5_dir
        self.h5_paths: List[str] = sorted(glob.glob(os.path.join(h5_dir, pattern)))
        if len(self.h5_paths) == 0:
            raise FileNotFoundError(f"No h5 files found in: {h5_dir} with pattern={pattern}")

        # global text -> (file_idx, local_id)
        self.text_to_loc: Dict[str, Tuple[int, int]] = {}

        # Per-process lazy file handles (NOT pickled safely across workers)
        self._files: Optional[List[h5py.File]] = None
        self._emb_dsets = None
        self._len_dsets = None

        # build global index (read unique_texts from each file once)
        self._build_global_index()

    def _build_global_index(self):
        for fi, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as f:
                texts = f["unique_texts"][:]
                for li, t in enumerate(texts):
                    if isinstance(t, bytes):
                        t = t.decode("utf-8")
                    # 如果同一个 text 在多个文件重复：默认保留第一次出现的
                    if t not in self.text_to_loc:
                        self.text_to_loc[t] = (fi, li)

    def _ensure_open(self):
        if self._files is not None:
            return
        self._files = [h5py.File(p, "r") for p in self.h5_paths]
        self._emb_dsets = [f["unique_embeddings"] for f in self._files]
        self._len_dsets = [f["text_valid_lengths"] if "text_valid_lengths" in f else None for f in self._files]

    def close(self):
        if self._files is None:
            return
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files = None
        self._emb_dsets = None
        self._len_dsets = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def contains(self, text: str) -> bool:
        return text in self.text_to_loc

    def get_location(self, text: str) -> Optional[Tuple[int, int]]:
        return self.text_to_loc.get(text)

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Return embedding np.ndarray (512,1024) if found else None
        """
        loc = self.text_to_loc.get(text)
        if loc is None:
            return None
        self._ensure_open()
        fi, li = loc
        return self._emb_dsets[fi][li]  # lazy slice read

    def get_embedding_and_len(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        loc = self.text_to_loc.get(text)
        if loc is None:
            return None, None
        self._ensure_open()
        fi, li = loc
        emb = self._emb_dsets[fi][li]
        vlen = None
        if self._len_dsets[fi] is not None:
            vlen = int(self._len_dsets[fi][li])
        return emb, vlen



# ---------------------- Example usage ----------------------
if __name__ == "__main__":
    # 1) 批量转换
    save_to_dir = "/path/to/save_to_dir"
    mapping = T5EmbeddingsH5.batch_convert_dir(save_to_dir, overwrite=False)
    print("Converted:", mapping)

    # 2) 打开某个 h5 做读取/追加
    one_h5 = next(iter(mapping.values()))
    with T5EmbeddingsH5(one_h5, mode="a") as store:
        print("N =", store.n_unique)

        # 构建索引并查 id
        store.build_text_index()
        some_text = "demo"
        idx = store.get_id(some_text)
        print("id =", idx)

        # 读取
        if idx is not None:
            t, emb, vlen = store.get_by_id(idx)
            print(t, emb.shape, vlen)

        # 追加（如果要去重：先 get_id 判断）
        # new_id = store.append_one("new text", np.zeros((512,1024), np.float16), 10)
        # print("new_id =", new_id)
