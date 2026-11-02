"""
ExamGuard - Dataset Preparation & Multi-Class Remapping Tool
Handles validation, train/val/test splitting, bounding box normalization verification,
and class ID remapping to build a unified ExamGuard dataset.
"""

import os
import shutil
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR

# Master Class Mapping for ExamGuard
MASTER_CLASS_NAMES = {
    0: "pencil",
    1: "eraser",
    2: "sharpener",
    3: "ruler",
    4: "mobile_phone",
    5: "book",
    6: "paper",
    7: "headphones"
}

def validate_yolo_label_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Checks that all bounding box coordinates in a YOLO label file are valid and normalized [0.0, 1.0]."""
    errors = []
    if not filepath.exists():
        return False, ["File not found"]
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {idx}: Expected 5 fields (<class> <x> <y> <w> <h>), got {len(parts)}")
            continue
        try:
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            for val_name, val in [("x", x), ("y", y), ("w", w), ("h", h)]:
                if not (0.0 <= val <= 1.05):  # allow small 5% float epsilon
                    errors.append(f"Line {idx}: {val_name}={val} is not normalized between 0.0 and 1.0")
        except ValueError as e:
            errors.append(f"Line {idx}: Parse error: {e}")

    return len(errors) == 0, errors

def remap_and_merge_dataset(
    source_images_dir: Path,
    source_labels_dir: Path,
    output_dir: Path,
    class_mapping: Dict[int, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42
):
    """
    Reads images and label text files from source, remaps class IDs according to class_mapping,
    and writes them into output_dir with images/train, images/valid, images/test.
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output structure
    for split in ["train", "valid", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Gather matching image and label pairs
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = [f for f in source_images_dir.glob("*.*") if f.suffix.lower() in valid_exts]
    print(f"[ExamGuard Dataset] Found {len(image_files)} images in {source_images_dir}")

    paired = []
    for img_path in image_files:
        lbl_path = source_labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            paired.append((img_path, lbl_path))

    print(f"[ExamGuard Dataset] Matched {len(paired)} image-label pairs.")
    random.shuffle(paired)

    n_total = len(paired)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": paired[:n_train],
        "valid": paired[n_train:n_train + n_val],
        "test": paired[n_train + n_val:]
    }

    total_remapped_boxes = 0

    for split_name, pairs in splits.items():
        dest_img_dir = output_dir / "images" / split_name
        dest_lbl_dir = output_dir / "labels" / split_name

        for img_src, lbl_src in pairs:
            # Copy image
            shutil.copy2(img_src, dest_img_dir / img_src.name)

            # Read, remap class IDs, write label
            remapped_lines = []
            with open(lbl_src, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        old_cls = int(parts[0])
                        new_cls = class_mapping.get(old_cls, old_cls)
                        remapped_lines.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                        total_remapped_boxes += 1

            with open(dest_lbl_dir / lbl_src.name, "w", encoding="utf-8") as f:
                f.writelines(remapped_lines)

    # Generate data.yaml
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": MASTER_CLASS_NAMES
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"[ExamGuard Dataset] Successfully generated unified dataset at {output_dir}")
    print(f"[ExamGuard Dataset] Split counts: Train={len(splits['train'])}, Val={len(splits['valid'])}, Test={len(splits['test'])}")
    print(f"[ExamGuard Dataset] Config saved to {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ExamGuard Dataset Preparation")
    parser.add_argument("--images", type=str, help="Source images folder")
    parser.add_argument("--labels", type=str, help="Source labels folder")
    parser.add_argument("--out", type=str, default="data/processed/ExamGuard_Unified_YOLO", help="Output directory")
    args = parser.parse_args()

    if args.images and args.labels:
        remap_and_merge_dataset(
            source_images_dir=Path(args.images),
            source_labels_dir=Path(args.labels),
            output_dir=Path(args.out),
            class_mapping={0: 0, 1: 1, 2: 2, 3: 3}
        )
    else:
        print("Usage: python src/prepare_dataset.py --images <path> --labels <path> --out <dest>")
