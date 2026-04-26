"""
Download and prepare the TACO (Trash Annotations in Context) dataset
for YOLOv8 fine-tuning.

TACO is a real-world litter/garbage dataset with 28 categories like
plastic bags, cigarette butts, wrappers, cans, bottles, etc.

Usage:
    python download_dataset.py

Outputs:
    dataset/
        images/train/   - training images
        images/val/     - validation images
        labels/train/   - YOLO-format .txt label files
        labels/val/
        data.yaml       - dataset config for YOLOv8
"""

import json
import os
import shutil
import urllib.request
from pathlib import Path
from collections import defaultdict
import random

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
TACO_DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/download.py"
)

# The 28 TACO supercategories we care about
# We collapse them all to a single "garbage" class for the simplest model,
# OR keep them separate for a richer model. Set SINGLE_CLASS = False to keep all.
SINGLE_CLASS = False

DATASET_DIR = Path("dataset")
VAL_SPLIT = 0.15  # 15% of images go to validation


def download_file(url: str, dest: Path):
    print(f"  Downloading {url.split('/')[-1]} ...")
    urllib.request.urlretrieve(url, dest)


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def build_dataset(annotations_path: Path):
    with open(annotations_path) as f:
        coco = json.load(f)

    # Build category map
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    supercats = {cat["id"]: cat.get("supercategory", cat["name"]) for cat in coco["categories"]}

    if SINGLE_CLASS:
        # All garbage → class 0
        class_names = ["garbage"]
        cat_to_class = {cid: 0 for cid in categories}
    else:
        # One class per supercategory
        unique_supers = sorted(set(supercats.values()))
        super_to_idx = {s: i for i, s in enumerate(unique_supers)}
        class_names = unique_supers
        cat_to_class = {cid: super_to_idx[supercats[cid]] for cid in categories}

    print(f"\nClasses ({len(class_names)}): {class_names}\n")

    # Group annotations by image
    img_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        img_annotations[ann["image_id"]].append(ann)

    images = coco["images"]
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_SPLIT))
    val_ids = {img["id"] for img in images[:n_val]}

    # Create directory structure
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = {"train": 0, "val": 0}

    for img_info in images:
        img_id = img_info["id"]
        img_file = img_info.get("file_name", "")
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_path = DATASET_DIR / "raw" / img_file
        if not src_path.exists():
            skipped += 1
            continue

        split = "val" if img_id in val_ids else "train"

        # Copy image
        dest_img = DATASET_DIR / "images" / split / Path(img_file).name
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_img)

        # Write label file
        label_lines = []
        for ann in img_annotations[img_id]:
            cls_idx = cat_to_class[ann["category_id"]]
            cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if nw > 0 and nh > 0:
                label_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not label_lines:
            continue

        label_path = DATASET_DIR / "labels" / split / (Path(img_file).stem + ".txt")
        label_path.write_text("\n".join(label_lines))
        written[split] += 1

    print(f"Written: train={written['train']}, val={written['val']}, skipped={skipped}")
    return class_names


def write_yaml(class_names):
    yaml_content = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Dataset config written to: {yaml_path}")
    return yaml_path


def main():
    print("=== TACO Dataset Setup ===\n")

    DATASET_DIR.mkdir(exist_ok=True)
    annotations_path = DATASET_DIR / "annotations.json"

    # Step 1: Download annotations
    if not annotations_path.exists():
        download_file(ANNOTATIONS_URL, annotations_path)
    else:
        print("  annotations.json already exists, skipping download.")

    # Step 2: Download images using TACO's own downloader
    raw_dir = DATASET_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    taco_dl = DATASET_DIR / "taco_download.py"

    if not taco_dl.exists():
        download_file(TACO_DOWNLOADER_URL, taco_dl)

    print("\nDownloading TACO images (this may take a while)...")
    print(f"Images will be saved to: {raw_dir}")
    ret = os.system(
        f"python {taco_dl} --dataset_path {raw_dir} --ann_file {annotations_path}"
    )
    if ret != 0:
        print("\nWARNING: Image download may have had errors.")
        print("If images are missing, run manually:")
        print(f"  python {taco_dl} --dataset_path {raw_dir} --ann_file {annotations_path}")

    # Step 3: Convert COCO annotations → YOLO format and split train/val
    print("\nConverting annotations to YOLO format...")
    class_names = build_dataset(annotations_path)

    # Step 4: Write data.yaml
    write_yaml(class_names)

    print("\nDone! Run train.py next.")


if __name__ == "__main__":
    main()
