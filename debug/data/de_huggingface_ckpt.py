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

        "2025-07-24/23-23-49",
        "2025-07-24/23-26-18",
        "2025-07-24/23-29-24",
        "2025-07-24/23-32-22",
        "2025-07-24/23-42-20",
        "2025-07-24/23-49-08",
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
    date_part, time_part = time_str.split('/')

    # 构造基础路径
    base_dir = Path.home() / "code/mdt24rss_fork/logs/runs" / date_part / time_part
    target_dir = Path.home() / "pretrained/mdt24rss"
    zip_filename = f"{date_part}_{time_part}.zip"
    zip_path = target_dir / zip_filename

    # 要压缩的目录列表
    dirs_to_compress = [
        ("checkpoints", base_dir / "checkpoints"),
        (".hydra", base_dir / ".hydra"),
    ]

    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)

    # 检查源目录是否存在
    for dir_name, source_dir in dirs_to_compress:
        if not source_dir.exists():
            print(f"Warning: {dir_name} directory not found: {source_dir}")
        else:
            print(f"Found {dir_name} dir: {source_dir}")

    print(f"Target zip file: {zip_path}")

    if force or not zip_path.exists():
        # 创建zip文件，使用ZIP_STORED（仅打包不压缩，速度快）
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
            # 遍历每个要压缩的目录
            for dir_name, source_dir in dirs_to_compress:
                if source_dir.exists():
                    print(f"Processing {dir_name}...")
                    # 遍历目录下的所有文件
                    for file_path in source_dir.rglob('*'):
                        if file_path.is_file():
                            # 计算在zip中的相对路径，保持目录结构
                            # 这样解压后会得到 date_part/time_part/checkpoints/... 和 date_part/time_part/.hydra/... 的结构
                            relative_path = Path(date_part) / time_part / dir_name / file_path.relative_to(source_dir)
                            zipf.write(file_path, relative_path)
                            print(f"Adding: {relative_path}")
                else:
                    print(f"Skipping {dir_name} (not found)")
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