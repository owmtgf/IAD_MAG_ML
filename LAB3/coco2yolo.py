import json
import yaml
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

seed = random.seed(42)


def make_yaml_file(cat_names: dict, output_folder: Path, images_folder: Path, split: str):
    yaml_dict = {
        "path": str(output_folder),
        "train": str(images_folder / "train"),
        "val": str(images_folder / "val"),
        "test": str(images_folder / "test"),
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

    make_yaml_file(categories_names, output_folder, images_path, split)

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

def train_val_split(yolo_path: Path, val_rate: float = 0.2):
    val_img_dir = yolo_path / "images" / "val"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir = yolo_path / "labels" / "val"
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_imgs = list((yolo_path / "images" / "train").glob("*.jpg"))
    train_labels = {f.stem: f for f in (yolo_path / "labels" / "train").glob("*.txt")}

    num_val_samples = int(len(train_imgs) * val_rate)
    val_images = set(random.sample(train_imgs, num_val_samples, seed=seed))
    for img_path in tqdm(val_images, desc="Creating val dataset"):
        label_path = train_labels.get(img_path.stem)
        if not label_path or not label_path.exists():
            print(f"Annotation for image {img_path.stem} not found! Skipping...")
            continue

        shutil.move(img_path, val_img_dir / img_path.name)
        shutil.move(label_path, val_lbl_dir / label_path.name)

    print(f"Moved {num_val_samples} images to validation set.")


if __name__ == "__main__":
    input_coco = Path("LAB3/data/dm-2026-lab-3-object-detection/usdc_train.json")
    output_folder = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO")
    images_path = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO/images/train")
    split = "train"

    convert(input_coco, output_folder, images_path, split)
    train_val_split(output_folder)