import os
from huggingface_hub import HfApi

api = HfApi()

# 上传文件到指定的 repo
upload_fns = [
    # "collected_data_0514_shovel_obj.zip",
    # "collected_data_0514_shovel_table.zip",
    # "0627_pot_source.patch01.zip",
    # "0627_pot_source_wP01.zip",

    # "0704_pepper_source.zip",
    # "0627_pot_light.zip",
    # "0627_pot_object.zip",

    # "0709_coffee_source.zip",

    # "0718_many_rand.zip"

    # "1009_spoon_pick_place.zip"

    "1010_sweep_bean.zip",
    "1010_pour_bean.zip",
    "1010_wipe_white_board.zip"
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