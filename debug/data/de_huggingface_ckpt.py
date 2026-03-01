import os
import zipfile
from pathlib import Path
from huggingface_hub import HfApi


def main():
    # 上传文件到指定的 repo
    upload_dirs = [
        # "2025-07-16/16-24-39",
        # "2025-07-16/17-15-42",
        # "2025-07-16/17-59-55",

        # "2025-07-21/11-20-27",
        # "2025-07-21/15-41-34",

        # "2025-07-22/17-06-07",

        # "2025-07-23/12-21-41",

        # "2025-07-23/19-54-04",
        # "2025-07-23/19-58-14",
        # "2025-07-23/20-03-27",
        # "2025-07-23/20-07-21",
        # "2025-07-23/20-09-31",

        # "2025-07-24/23-23-49",
        # "2025-07-24/23-26-18",
        # "2025-07-24/23-29-24",
        # "2025-07-24/23-32-22",
        # "2025-07-24/23-42-20",
        # "2025-07-24/23-49-08",

        "cospred2_2b_force_tcl_2025-11-04_16-25-46",
    ]

    for upload_dir in upload_dirs:
        compress_and_upload_checkpoint(
            upload_dir,
            repo_id="ygtxr1997/tcl_models",
            repo_type="model",
            force=True,  # True: recover existing .zip file
        )


def compress_and_upload_checkpoint(time_str, repo_id, repo_type="model", force: bool = True):
    """
    压缩checkpoints和.hydra文件夹并上传到HuggingFace

    Args:
        time_str: 形如 "2025-07-16/20-17-52" 的时间字符串
        repo_id: HuggingFace repo ID，如 "username/repo_name"
        repo_type: "model" 或 "dataset"
    """

    # 解析时间字符串
    if "/" in time_str:
        date_part, time_part = time_str.split('/')
    else:
        date_part = time_str
        time_part = ""

    # 构造基础路径
    base_dir = Path.home() / "code/cospred2nvidia/checkpoints/cosmos_predict2/debug" / date_part / time_part
    target_dir = Path.home() / "pretrained/cospred2nvidia"
    zip_filename = f"{date_part}_{time_part}.zip" if time_part else f"{date_part}.zip"
    zip_path = target_dir / zip_filename

    # 要压缩的内容：目录 + 单个文件
    items_to_compress = [
        ("checkpoints", base_dir / "checkpoints", "dir"),
        ("config.pkl", base_dir / "config.pkl", "file"),
        ("config.yaml", base_dir / "config.yaml", "file"),
        ("DeviceMonitor", base_dir / "DeviceMonitor", "dir"),
        ("statistics.json", base_dir / "statistics.json", "file"),
        ("stdout.log", base_dir / "stdout.log", "file"),
    ]  # (name, path, type)

    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)
    # 检查源是否存在
    for name, source_path, item_type in items_to_compress:
        if not source_path.exists():
            print(f"Warning: {name} not found: {source_path}")
        else:
            print(f"Found {name}: {source_path}")

    print(f"Target zip file: {zip_path}")
    if force or not zip_path.exists():
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
            for name, source_path, item_type in items_to_compress:
                if not source_path.exists():
                    print(f"Skipping {name} (not found)")
                    continue

                if item_type == "file":
                    # 添加单个文件
                    relative_path = Path(time_str) / name
                    zipf.write(source_path, relative_path)
                    print(f"Adding file: {relative_path}")
                elif item_type == "dir":
                    # 添加目录及其所有内容
                    print(f"Processing {name}...")
                    for file_path in source_path.rglob('*'):
                        if file_path.is_file():
                            relative_path = Path(time_str) / name / file_path.relative_to(source_path)
                            zipf.write(file_path, relative_path)
                            print(f"Adding: {relative_path}")
    else:
        print(f"Zip file already exists: {zip_path}")

    print(f"Zip finished: {zip_path}")
    print(f"File size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")

    # 上传到HuggingFace
    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        print(f"Uploading {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=zip_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"Uploaded! File: {zip_filename}")

        # 生成下载链接
        download_url = f"https://huggingface.co/{repo_id}/resolve/main/{zip_filename}"
        print(f"Download Url: {download_url}")

    except Exception as e:
        print(f"Uploading failed: {e}")
        raise


def download_and_extract_example():
    """
    示例：下载并解压的代码
    """
    example_code = '''
    # Run this on the other computer
    import zipfile
    from pathlib import Path

    # After downloading to: ~/downloads/2025-07-16_20-17-52.zip
    zip_path = Path.home() / "downloads/2025-07-16_20-17-52.zip"
    extract_to = Path.home() / "downloads"

    # Unzip
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)

    # Unzip directory will be：
    # ~/downloads/2025-07-16/20-17-52/checkpoints/
    # ~/downloads/2025-07-16/20-17-52/.hydra/
    print("Unzip finished！")
    print(f"checkpoints position: {extract_to}/2025-07-16/20-17-52/checkpoints")
    print(f".hydra position: {extract_to}/2025-07-16/20-17-52/.hydra")
    '''
    print("=" * 50)
    print("Downloading example：")
    print("=" * 50)
    print(example_code)


if __name__ == "__main__":
    main()