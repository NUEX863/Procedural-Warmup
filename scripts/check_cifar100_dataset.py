import hashlib
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# 你的 CIFAR-100 解压目录
DATA_DIR = Path(
    r"D:\01projects\01research\01paper\paper01\data\cifar100\cifar-100-python"
)

# 官方文件的 MD5 校验值
EXPECTED_MD5 = {
    "train": "16019d7e3df5f24257cddd939b257f8d",
    "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
    "meta": "7973b15100ade9c7d40fb424638fde48",
}

EXPECTED_COUNTS = {
    "train": 50000,
    "test": 10000,
}


def calculate_md5(path):
    """计算文件的 MD5 值。"""
    digest = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_pickle(path):
    """读取 CIFAR-100 官方 Python 格式文件。"""
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def get_value(data, key):
    """兼容字节类型和字符串类型的字典键。"""
    if key.encode() in data:
        return data[key.encode()]
    if key in data:
        return data[key]
    raise KeyError(f"Missing key: {key}")


def main():
    print("=" * 60)
    print("CIFAR-100 Dataset Integrity Check")
    print(f"Dataset path: {DATA_DIR}")
    print("=" * 60)

    errors = []

    # 1. 检查目录
    if not DATA_DIR.is_dir():
        print("[FAIL] Dataset directory does not exist.")
        sys.exit(1)

    # 2. 检查文件是否存在、MD5 是否与官方一致
    for filename, expected_md5 in EXPECTED_MD5.items():
        path = DATA_DIR / filename

        if not path.is_file():
            errors.append(f"Missing file: {filename}")
            print(f"[FAIL] Missing file: {path}")
            continue

        actual_md5 = calculate_md5(path)
        size_mb = path.stat().st_size / (1024 * 1024)

        if actual_md5 == expected_md5:
            print(f"[PASS] {filename}: MD5 verified, {size_mb:.2f} MB")
        else:
            errors.append(f"MD5 mismatch: {filename}")
            print(f"[FAIL] {filename}: MD5 mismatch")
            print(f"       Expected: {expected_md5}")
            print(f"       Actual:   {actual_md5}")

    if errors:
        print("\n[FAIL] File integrity check failed.")
        print("\n".join(errors))
        sys.exit(1)

    # 3. 读取训练集和测试集
    train = load_pickle(DATA_DIR / "train")
    test = load_pickle(DATA_DIR / "test")
    meta = load_pickle(DATA_DIR / "meta")

    # 4. 检查图片数量、尺寸和数据类型
    for name, dataset in [("train", train), ("test", test)]:
        images = np.asarray(get_value(dataset, "data"))
        labels = get_value(dataset, "fine_labels")

        expected_count = EXPECTED_COUNTS[name]

        if len(images) != expected_count:
            errors.append(
                f"{name}: expected {expected_count} images, "
                f"found {len(images)}"
            )

        if images.shape != (expected_count, 3072):
            errors.append(
                f"{name}: unexpected image array shape {images.shape}"
            )

        if images.dtype != np.uint8:
            errors.append(
                f"{name}: expected uint8 image data, got {images.dtype}"
            )

        if len(labels) != expected_count:
            errors.append(
                f"{name}: label count {len(labels)} does not match image count"
            )
            continue

        labels = [int(x) for x in labels]

        if not all(0 <= x < 100 for x in labels):
            errors.append(f"{name}: labels outside range 0-99")

        counts = Counter(labels)

        expected_per_class = 500 if name == "train" else 100

        if len(counts) != 100:
            errors.append(f"{name}: expected 100 fine classes")

        if any(counts[i] != expected_per_class for i in range(100)):
            errors.append(
                f"{name}: unexpected fine-class distribution"
            )

        print(
            f"[{'PASS' if not any(name in e for e in errors) else 'FAIL'}] "
            f"{name}: images={len(images):,}, "
            f"shape={images.shape}, "
            f"classes={len(counts)}"
        )

    # 5. 检查类别名称
    fine_names = get_value(meta, "fine_label_names")
    coarse_names = get_value(meta, "coarse_label_names")

    print(f"Fine classes: {len(fine_names)}")
    print(f"Coarse classes: {len(coarse_names)}")

    if len(fine_names) != 100:
        errors.append("Expected 100 fine class names")

    if len(coarse_names) != 20:
        errors.append("Expected 20 coarse class names")

    # 6. 最终结论
    print("\n" + "=" * 60)

    if errors:
        print("[FAIL] Dataset verification failed:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("[SUCCESS] CIFAR-100 dataset is complete and verified.")
    print("Training images: 50,000")
    print("Test images:     10,000")
    print("Fine classes:    100")
    print("Coarse classes:  20")
    print("=" * 60)


if __name__ == "__main__":
    main()