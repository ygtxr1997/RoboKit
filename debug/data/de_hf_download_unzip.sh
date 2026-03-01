#!/bin/bash

# 设置 Repo 和文件信息
repo_id="ygtxr1997/tcl_models"  # 替换为你的 Hugging Face Repo
download_fns=(
  "2025-07-23_19-54-04.zip"
  "2025-07-23_19-58-14.zip"
  "2025-07-23_20-03-27.zip"
  "2025-07-23_20-07-21.zip"
  "2025-07-23_20-09-31.zip"
  "2025-07-24_23-23-49.zip"
  "2025-07-24_23-26-18.zip"
  "2025-07-24_23-29-24.zip"
  "2025-07-24_23-32-22.zip"
  "2025-07-24_23-42-20.zip"
  "2025-07-24_23-49-08.zip"
)  # 你要下载的文件名列表
data_root="/home/geyuan/Documents/code/mdt24rss_fork/logs/runs/"  # 下载目录

# 创建下载目录
mkdir -p "$data_root"

# 下载文件并解压
for file in "${download_fns[@]}"; do
    # 获取文件的下载链接
    file_url="https://huggingface.co/$repo_id/resolve/main/$file"
    file_path="$data_root/$file"

    # 使用 wget 下载文件 - 确保前台同步执行
    echo "正在下载 $file ..."

    if wget --no-background --progress=bar:force --timeout=300 --tries=3 -c "$file_url" -O "$file_path"; then
        echo "$file 下载成功"
    else
        echo "错误: $file 下载失败"
        continue
    fi

    # 验证下载的文件
    if [[ ! -f "$file_path" ]] || [[ ! -s "$file_path" ]]; then
        echo "错误: 文件 $file 下载不完整或不存在"
        continue
    fi

    # 检查下载文件是否是 .zip 格式
    if [[ "$file" == *.zip ]]; then
        # 解压 .zip 文件前验证文件完整性
        echo "正在验证并解压 $file ..."

        # 使用 unzip -t 测试 zip 文件完整性
        if unzip -t "$file_path" >/dev/null 2>&1; then
            echo "ZIP 文件完整性验证通过"
            if unzip -q "$file_path" -d "$data_root"; then
                echo "$file 解压完成"
            else
                echo "错误: $file 解压失败"
                continue
            fi
        else
            echo "错误: $file ZIP文件损坏或不完整"
            continue
        fi
    fi

    echo "$file 处理完成，存储路径: $data_root"
    echo "----------------------------------------"
done

echo "所有文件下载和解压完成！"