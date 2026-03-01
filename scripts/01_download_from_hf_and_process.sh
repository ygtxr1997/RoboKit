#!/bin/bash

# 错误处理和调试
set -e
#set -x

# 配置变量
FILE_NAMES=(
#  "0627_pot_source"
#  "0627_pot_light"
#  "0627_pot_object"
#  "0704_pepper_source"
#  "0709_coffee_source"
#  "0709_coffee_distractor_rand"
#  "0709_coffee_table_rand"
#  "0709_coffee_object_rand"
#  "0709_coffee_light_rand"
#  "0704_pepper_distractor_rand"
#  "0704_pepper_table_rand"
#  "0704_pepper_object_rand"
#  "0704_pepper_light_rand"
#  "0627_pot_table_rand"
#  "1009_spoon_pick_place"
#  "1021_sweep_bean"
#"1024_sweep_bean"
#"1024_eggs_pick_place"
#"1024_pour_water"
#"1024_wipe_white_board"
#"1117_red_peper_pick_place"
#"1117_carrot_pick_place"
#"1201_wipe_blackboard"
#"1201_pour_water"
#"1201_banana"
#"1201_pepper"
#"1201_pot"
#"1201_coffee"
#"1201_screw_bulb"
"1201_screw_bulb_turn_off"
#"0209_tower_boby"
)

# 路径配置
LOCAL_HOME="/home/geyuan"
DATASETS_DIR="${LOCAL_HOME}/datasets/TCL"
LOCAL_SOFT_DIR="${LOCAL_HOME}/local_soft/TCL/"
HDF5_DIR="${LOCAL_HOME}/local_soft/TCL/hdf5"


# 创建必要的目录
mkdir -p "${DATASETS_DIR}" "${LOCAL_SOFT_DIR}" "${HDF5_DIR}"

# 函数：检查并下载文件
download_if_needed() {
    local zip_path="$1"  # "${DATASETS_DIR}/${ZIP_FILE}"
    local download_dir="$2"

    # ZIP 文件存在, 跳过
    if [[ -f "${zip_path}" ]]; then
        echo "ZIP文件已存在: ${zip_path}"
        return 0
    fi

    # 目标解压目录存在, 跳过
    local unzip_dir="${DATASETS_DIR}/${FILE_NAME}/"
    if [[ -e "${unzip_dir}" ]]; then
        echo "ZIP解压目录已存在: ${unzip_dir}"
        return 0
    fi

    echo "ZIP文件不存在，开始下载..."
    cd "${download_dir}"

    # 下载文件，支持断点续传
    if ! wget -c "${HF_DOWNLOAD_URL}"; then
        echo "错误: 下载失败"
        return 1
    fi

    # 验证下载的文件
    if [[ ! -f "${ZIP_FILE}" ]]; then
        echo "错误: 下载的文件不存在"
        return 1
    fi

    cd - > /dev/null
    echo "下载完成: ${zip_path}"
}

# 函数：检查并解压文件
extract_if_needed() {
    local zip_path="$1"
    local extract_dir="$2"
    local expected_folder="$3"

    echo "Checking zip state..."

    # 检查是否已经解压（支持软链接的智能检查）
    local target_path="${extract_dir}/${expected_folder}"

    # 使用 -e 检查路径是否存在（无论是实际目录还是软链接）
    if [[ -e "${target_path}" ]]; then
        if [[ -L "${extract_dir}" ]]; then
            # 如果父目录是软链接，解析真实路径
            local real_extract_dir=$(readlink -f "${extract_dir}")
            local real_target_path="${real_extract_dir}/${expected_folder}"
            echo "Checking soft link："
            echo "  soft link dir: ${extract_dir} -> ${real_extract_dir}"
            echo "  target dir: ${real_target_path}"
            if [[ -e "${real_target_path}" ]]; then
                echo "skip unzip, folder exists in real path: ${real_target_path}"
                return 0
            fi
        else
            echo "Dir exists: ${target_path}"
            echo "Skip unzip"
            return 0
        fi
    fi

    # 额外检查：在软链接环境中，可能需要检查实际路径
    if [[ -L "${extract_dir}" ]]; then
        local real_extract_dir=$(readlink -f "${extract_dir}")
        local real_target_path="${real_extract_dir}/${expected_folder}"
        if [[ -e "${real_target_path}" ]]; then
            echo "Dir exists: ${real_target_path}"
            echo "Skip unzip, folder exists in real path"
            return 0
        fi
    fi

    echo "Start unzip: ${extract_dir}"
    cd "${extract_dir}"

        # 检查ZIP文件是否存在
    if [[ ! -f "${zip_path}" ]]; then
        echo "Error: ZIP file not exists: ${zip_path}"
        return 1
    fi

    # 首先查看ZIP文件内容，了解其结构
    echo "ZIP file preview："
    unzip -l "${zip_path}" | head -10

    # 解压文件并显示进度
    echo "Unzip..."
    if ! unzip -q "${zip_path}"; then
        echo "Error: Unzip failed"
        cd - > /dev/null
        return 1
    fi

    cd - > /dev/null

    # 显示解压后的内容
    echo "Unzip succeed. Extracted content preview:"
    ls -la "${extract_dir}" | grep -E "^d.*$(date +%Y|%m|%d)" || ls -la "${extract_dir}" | head -10

    # 尝试找到实际的文件夹名称
    local actual_folder=$(find "${extract_dir}" -maxdepth 1 -name "${expected_folder}*" -type d 2>/dev/null | head -1)
    if [[ -n "${actual_folder}" ]]; then
        echo "Found target dir: ${actual_folder}"
    else
        echo "Warning: didn't find expected folder: ${expected_folder}"
        echo "Current dir:"
        ls -la "${extract_dir}"
    fi
}

# 函数：安全复制文件
copy_file_if_needed() {
    local src="$1"
    local dst="$2"

    if [[ -f "${dst}" ]]; then
        echo "File exists: ${dst}"
        return 0
    fi

    if [[ -e "${LOCAL_SOFT_DIR}/${FILE_NAME}/" ]]; then
        echo "File exists: ${dst}"
        return 0
    fi

    echo "Copy: ${src} -> ${dst}"

    if ! cp "${src}" "${dst}"; then
        echo "Error: copy file failed"
        return 1
    fi

    echo "File copied successfully: ${dst}"
}

# 主流程
main() {
  echo "Start processing: ${FILE_NAME}"

  for FILE_NAME in "${FILE_NAMES[@]}"; do
    ZIP_FILE="${FILE_NAME}.zip"
    HF_DOWNLOAD_URL="https://huggingface.co/datasets/ygtxr1997/tcl_inovo/resolve/main/${ZIP_FILE}"

    # 步骤1: 检查并下载到datasets目录
    local datasets_zip="${DATASETS_DIR}/${ZIP_FILE}"
    download_if_needed "${datasets_zip}" "${DATASETS_DIR}"

    # 步骤2: 检查并解压到datasets目录
    extract_if_needed "${datasets_zip}" "${DATASETS_DIR}" "${FILE_NAME}"

    # 步骤3: 复制到local_soft目录
    local local_soft_zip="${LOCAL_SOFT_DIR}/${ZIP_FILE}"
    copy_file_if_needed "${datasets_zip}" "${local_soft_zip}"

    # 步骤4: 检查并解压到local_soft目录
    extract_if_needed "${local_soft_zip}" "${LOCAL_SOFT_DIR}" "${FILE_NAME}"

    # 步骤5: 运行数据预处理
    echo "Start dataset convert..."
    local source_dir="${LOCAL_SOFT_DIR}/${FILE_NAME}"
    local output_file="${HDF5_DIR}/${FILE_NAME}_240p.h5"

    if [[ ! -d "${source_dir}" ]]; then
        echo "Error: source dir doesn't exist: ${source_dir}"
        exit 1
    fi

    python -m robokit.datasets.preprocess_data \
        -R "${source_dir}" \
        --as_hdf5 "${output_file}"
  done

  echo "Process successfully completed."
}

# 执行主函数
main "$@"