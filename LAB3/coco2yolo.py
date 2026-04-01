import json
import yaml
import random
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

np.random.seed(42)


def make_yaml_file(cat_names: dict, output_folder: Path):
    yaml_dict = {
        "path": str(output_folder),
        "train": str("images/train"),
        "val": str("images/val"),
        "test": str("images/test"),
        "names": cat_names,
    }
    with open(output_folder / "yolo_dataset.yaml", "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)


def convert(input_coco: Path, output_folder: Path, images_path: Path, split: str = "train"):
    output_folder.mkdir(parents=True, exist_ok=True)

    ann_folder = output_folder / "labels" / split
    ann_folder.mkdir(parents=True, exist_ok=True)

    images_names = [img.name for img in images_path.glob("*.jpg")]

    with open(input_coco, "r") as f:
        coco = json.load(f)
    
    images = {img["id"]: img for img in coco["images"]}
    categories_names = {int(cat["id"]): cat["name"] for cat in coco["categories"]}

    cats_mapping = {idx: idx for idx in categories_names}
    if min(categories_names.keys()) >= 1:
        cats_mapping = {cat_id: i for i, cat_id in enumerate(categories_names.keys())}

    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    for img_id, anns in tqdm(img_anns.items(), desc="Processing COCO annotations"):
        img = images.get(img_id)
        if not img:
            continue

        if img["file_name"] not in images_names:
            print(f"Image {img['file_name']} does not exist in images folder! Skipping...")
            continue

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]

            img_width = img["width"]
            img_height = img["height"]

            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            w = w / img_width
            h = h / img_height

            yolo_cat = cats_mapping[ann["category_id"]]
            lines.append(f"{yolo_cat} {x_center} {y_center} {w} {h}")

        with open((ann_folder / img["file_name"]).with_suffix(".txt"), "w") as f:
            f.write("\n".join(lines))

    print("Done!")
    return categories_names

def train_val_split(
        yolo_path: Path,
        input_train_images_path: Path,
        test_images_path: Path,
        categories: dict,
        val_rate: float = 0.06,
    ):
    val_img_dir = yolo_path / "images" / "val"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    
    train_img_dir = output_folder / "images" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)

    test_img_dir = output_folder / "images" / "test"
    test_img_dir.mkdir(parents=True, exist_ok=True)

    val_lbl_dir = yolo_path / "labels" / "val"
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    input_imgs = sorted(set(input_train_images_path.glob("*.jpg"))) + sorted(set(input_train_images_path.glob("*.png")))
    input_labels = {f.stem: f for f in (yolo_path / "labels" / "train").glob("*.txt")}

    num_val_samples = int(len(input_imgs) * val_rate)
    val_images = set(np.random.choice(input_imgs, size=num_val_samples, replace=False))

    print(f"Overall we have {len(input_imgs)} images and {len(input_labels)} labels")
    print(f"Number of validation images: {num_val_samples}")
    skipped = 0
    for img_path in tqdm(val_images, desc="Creating val dataset"):
        label_path = input_labels.get(img_path.stem)
        if not label_path or not label_path.exists():
            print(f"Annotation for image {img_path.stem} not found! Skipping...")
            skipped += 1
            continue

        val_img_out_path: Path = val_img_dir / img_path.name
        val_img_out_path.symlink_to(img_path.resolve())
        shutil.move(label_path, val_lbl_dir / label_path.name)
    
    print(f"Skipped {skipped} labels")
    print(f"Moved {num_val_samples} labels to validation set.")

    train_set = set(input_imgs) - val_images
    print(f"Creating train images set of {len(train_set)} images")
    for img_path in tqdm(train_set):
        train_img_out_path: Path = train_img_dir / img_path.name
        train_img_out_path.symlink_to(img_path.resolve())

    test_imgs = sorted(set(test_images_path.glob("*.jpg"))) + sorted(set(test_images_path.glob("*.png")))
    print(f"Creating test images set of {len(test_imgs)} images")
    for img_path in tqdm(test_imgs):
        test_img_out_path: Path = test_img_dir / img_path.name
        test_img_out_path.symlink_to(img_path.resolve())
    
    make_yaml_file(categories, output_folder)


if __name__ == "__main__":
    input_coco = Path("./data/dm-2026-lab-3-object-detection/usdc_train.json")  # path to coco annotations
    output_folder = Path("./data/dm-2026-lab-3-object-detection/YOLO")
    train_images_path = Path("./data/dm-2026-lab-3-object-detection/train_images/train_images")  # path to train images
    test_images_path = Path("./data/dm-2026-lab-3-object-detection/test_images/test_images")  # path to test images
    split = "train"

    categories = convert(input_coco, output_folder, train_images_path, split)
    train_val_split(output_folder, train_images_path, test_images_path, categories)