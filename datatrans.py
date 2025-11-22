import os
import json
import random
from pathlib import Path
from PIL import Image

# ==========================================================
# 你只需要修改下面两个路径
# 原始数据集根目录（里面包含 ds/ann 和 ds/img）
SRC_ROOT = Path(r"C:\Users\user\PycharmProjects\LightYolo\datatest\dataorigin")

# YOLO 输出数据集目录
DST_ROOT = Path(r"C:\Users\user\PycharmProjects\LightYolo\datatest\datanew")
# ==========================================================

ANN_DIR = SRC_ROOT / "ds" / "ann"
IMG_DIR = SRC_ROOT / "ds" / "img"

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15        # 剩下 0.15 是 test
OUT_IMG_EXT = "jpg"      # 输出图片格式，可改成 png

random.seed(42)


def read_all_ann_files():
    ann_files = sorted(ANN_DIR.glob("*.json"))
    if not ann_files:
        raise RuntimeError(f"在 {ANN_DIR} 找不到任何 .json 标注文件！")
    return ann_files


def collect_classes(ann_files):
    class_names = set()
    for ann_path in ann_files:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data.get("objects", []):
            title = obj.get("classTitle")
            if title:
                class_names.add(title)

    classes = sorted(class_names)
    print("\n=== 检测到的类别及其 ID ===")
    for i, name in enumerate(classes):
        print(f"{i}: {name}")
    print("=========================\n")
    return classes


def split_dataset(ann_files):
    n = len(ann_files)
    idx = list(range(n))
    random.shuffle(idx)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_idx = set(idx[:n_train])
    val_idx = set(idx[n_train:n_train+n_val])

    split_map = {}
    for i, ann_path in enumerate(ann_files):
        if i in train_idx:
            split_map[ann_path] = "train"
        elif i in val_idx:
            split_map[ann_path] = "val"
        else:
            split_map[ann_path] = "test"

    print(f"总数量: {n}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
    return split_map


def ensure_dirs():
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        d = DST_ROOT / sub
        d.mkdir(parents=True, exist_ok=True)


def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert_one(ann_path, split, class2id):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_name = data.get("name") or ann_path.name.replace(".json", "")
    img_stem = Path(img_name).stem
    src_img_path = IMG_DIR / img_name

    if not src_img_path.is_file():
        print(f"[跳过] 找不到图片: {src_img_path}")
        return

    size = data.get("size", {})
    W = size.get("width")
    H = size.get("height")

    if W is None or H is None:
        with Image.open(src_img_path) as im:
            W, H = im.size

    objects = data.get("objects", [])
    yolo_lines = []

    for obj in objects:
        cls_name = obj.get("classTitle")
        if cls_name not in class2id:
            continue
        cls_id = class2id[cls_name]

        points = obj.get("points", {})
        exterior = points.get("exterior")
        if not exterior:
            continue

        x1, y1, x2, y2 = polygon_to_bbox(exterior)

        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        x_c = (x1 + x2) / 2 / W
        y_c = (y1 + y2) / 2 / H
        w = bw / W
        h = bh / H

        yolo_lines.append(
            f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
        )

    # 保存图片
    dst_img = DST_ROOT / "images" / split / f"{img_stem}.{OUT_IMG_EXT}"
    if not dst_img.is_file():
        with Image.open(src_img_path) as im:
            fmt = "JPEG" if OUT_IMG_EXT.lower() == "jpg" else OUT_IMG_EXT.upper()
            im.convert("RGB").save(dst_img, fmt)

    # 保存标签
    dst_label = DST_ROOT / "labels" / split / f"{img_stem}.txt"
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


def write_yaml(classes):
    yaml_path = DST_ROOT / "dataset.yaml"
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:"
    ]
    for i, name in enumerate(classes):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n已生成 YOLOv8 配置文件: {yaml_path}")


def main():
    ann_files = read_all_ann_files()
    classes = collect_classes(ann_files)
    class2id = {c: i for i, c in enumerate(classes)}

    split_map = split_dataset(ann_files)
    ensure_dirs()

    for ann_path, split in split_map.items():
        convert_one(ann_path, split, class2id)

    write_yaml(classes)
    print("\n转换完成！YOLO 数据集已生成于：")
    print(DST_ROOT.resolve())


if __name__ == "__main__":
    main()
