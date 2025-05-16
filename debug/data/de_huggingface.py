import os
from huggingface_hub import HfApi

api = HfApi()

# 上传文件到指定的 repo
upload_fns = [
    "collected_data_0514_shovel_obj.zip",
    "collected_data_0514_shovel_table.zip",
]

for upload_fn in upload_fns:
    data_root = "/home/geyuan/datasets/TCL"
    repo_id = "ygtxr1997/tcl_inovo"  # 替换为你的 Hugging Face Repo
    file_path = os.path.join(data_root, upload_fn)  # 大文件路径

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=upload_fn,  # 在 repo 中存储的文件名
        repo_id=repo_id,
        repo_type="dataset",  # 或 "model" 根据上传类型选择
    )