
from pathlib import Path
import hashlib
import os
import shutil

from huggingface_hub import hf_hub_download
from torchvision.datasets import SVHN


# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 最终数据保存路径：data/svhn/，没有 raw 文件夹
DATA_DIR = ROOT / "data" / "svhn"

# Hugging Face 上提供原始 MAT 文件的镜像仓库
REPO_ID = "darcook/DCPV-Artifacts"

# TorchVision 官方使用的 MD5 校验值
FILES = {
    "train": {
        "name": "train_32x32.mat",
        "md5": "e26dedcc434d2e4c54c9b2d4a06d8373",
        "count": 73257,
    },
    "test": {
        "name": "test_32x32.mat",
        "md5": "eb5a983be6a315427106f1b164d9cef3",
        "count": 26032,
    },
}


def calculate_md5(path):
    """计算文件的 MD5 指纹。"""
    digest = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_file(path, expected_md5):
    """检查文件是否存在、内容是否与官方文件一致。"""
    return (
        path.is_file()
        and calculate_md5(path) == expected_md5
    )


def download_svhn_file(split):
    """从 Hugging Face 下载原始 SVHN MAT 文件。"""

    info = FILES[split]
    filename = info["name"]
    expected_md5 = info["md5"]

    destination = DATA_DIR / filename

    # 已有正确文件，不重复下载
    if verify_file(destination, expected_md5):
        print(f"[OK] 已存在且校验通过: {filename}")
        return

    print(f"\n[DOWNLOAD] 正在下载 {filename}")

    # Hugging Face 缓存下载
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/svhn/{filename}",
    )

    source = Path(downloaded_path)

    # 先验证下载结果
    actual_md5 = calculate_md5(source)

    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"{filename} 校验失败！\n"
            f"官方 MD5: {expected_md5}\n"
            f"下载 MD5: {actual_md5}\n"
            "文件可能不是官方原始版本，已停止操作。"
        )

    # 验证通过后才复制到项目目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = DATA_DIR / (filename + ".tmp")

    try:
        shutil.copyfile(source, temp_path)

        if not verify_file(temp_path, expected_md5):
            raise RuntimeError(f"{filename} 复制后校验失败")

        os.replace(temp_path, destination)

    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"[OK] 下载成功: {destination}")
    print(f"[OK] MD5: {expected_md5}")


def main():
    print("=" * 60)
    print("SVHN Dataset Download - Hugging Face")
    print(f"保存目录: {DATA_DIR}")
    print("=" * 60)

    # 下载完整训练集和测试集
    for split in ("train", "test"):
        download_svhn_file(split)

    print("\n开始检查 TorchVision 是否能够读取数据...")

    # 确认现有训练项目可直接读取
    for split in ("train", "test"):
        dataset = SVHN(
            root=str(DATA_DIR),
            split=split,
            download=False,
        )

        expected_count = FILES[split]["count"]

        if len(dataset) != expected_count:
            raise RuntimeError(
                f"{split} 图片数量错误: "
                f"实际 {len(dataset)}, "
                f"应为 {expected_count}"
            )

        if dataset.data.shape != (
            expected_count, 3, 32, 32
        ):
            raise RuntimeError(
                f"{split} 图片维度错误: "
                f"{dataset.data.shape}"
            )

        print(
            f"[PASS] {split.upper()}: "
            f"{len(dataset):,} 张图片，"
            f"形状 {dataset.data.shape}"
        )

    print("\n" + "=" * 60)
    print("[SUCCESS] SVHN 下载和验证全部完成")
    print(f"保存路径: {DATA_DIR}")
    print("没有划分子集，没有更改标签。")
    print("=" * 60)


if __name__ == "__main__":
    main()
