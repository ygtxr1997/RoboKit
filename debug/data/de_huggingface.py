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

    # "1017_sweep_bean.zip",
    # "1024_sweep_bean.zip",
    # "1024_eggs_pick_place.zip",
    # "1024_pour_water.zip",
    # "1024_wipe_white_board.zip"
    # "1117_red_peper_pick_place.zip",
    # "1117_carrot_pick_place.zip",

    # "1201_wipe_blackboard.zip",
    # "1201_pour_water.zip",
    # "1201_banana.zip",
    # "1201_pepper.zip",
    # "1201_pot.zip",
    # "1201_coffee.zip",

    # "1201_screw_bulb.zip",
    # "1201_screw_bulb_turn_off.zip",
    # "1201_screw_bulb_turn_off_table.zip",
    # "1201_screw_bulb_turn_off_object.zip",
    # "1201_screw_bulb_turn_off_light.zip",

    # "0209_tower_boby.zip",
    # "0209_tower_boby_hard.zip",
    # "0209_tower_boby_easy.zip",

    # "0417_put_mouse.zip"
    # "0417_ethernet.zip"
    # "0417_greenyellowred.zip"
    # "0417_test_tube.zip",
    # "0417_french_press.zip",
    # "0417_brush.zip",
    # "0417_plastic_egg.zip",
    "0417_put_mouse_hard.zip",

    # "260514_ethernet.zip",
    # "0731_disk_force_3.zip",
    # "0731_disk_force_6.zip",
    # "0731_disk_force_7.zip",
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