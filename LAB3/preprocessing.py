from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset as TorchDataset

from utils import yaml_read, compute_iou


@dataclass
class Bbox:
    cat_id: int
    cat_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self) -> tuple[float]:
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return x1, y1, x2, y2


@dataclass
class ImageData:
    image_path: Path
    label_path: Path
    bboxes: list[Bbox] = field(default_factory=list)


@dataclass
class Dataset:
    images: list[ImageData] = field(default_factory=list)

    def count_boxes(self):
        return sum(len(img.bboxes) for img in self.images)
    

def read_yolo_dataset(images_dir: Path, labels_dir: Path, yaml_path: Path) -> Dataset:
    dataset = Dataset()
    category_names = yaml_read(yaml_path)

    skipped_images = 0
    for img_path in images_dir.glob("*.jpg"):
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f"[WARNING] No label file for image {img_path.name}. Skipping...")
            skipped_images += 1
            continue

        bboxes = []
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cat_id, x, y, w, h = parts
            cat_name = category_names.get(int(cat_id))
            if not cat_name:
                print(f"[WARNING] Category name for id {cat_id} not found! Using id as name.")
                cat_name = str(cat_id)

            bbox = Bbox(
                cat_id=int(cat_id),
                cat_name=cat_name,
                x_center=float(x),
                y_center=float(y),
                width=float(w),
                height=float(h),
            )
            bboxes.append(bbox)

        dataset.images.append(
            ImageData(
                image_path=img_path,
                label_path=label_path,
                bboxes=bboxes,
            )
        )
    print(f"Skipped {skipped_images} images while reading dataset")

    return dataset


def filter_duplicate_bboxes(
    bboxes: list[Bbox],
    iou_threshold: float = 0.9
) -> list[Bbox]:
    filtered = []
    for bbox in bboxes:
        is_duplicate = False

        for kept in filtered:
            if bbox.cat_id != kept.cat_id:
                continue

            if compute_iou(bbox.to_xyxy(), kept.to_xyxy()) >= iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(bbox)

    return filtered


def preprocess_dataset(dataset: Dataset, iou_threshold=0.9):
    # Bboxes filtering
    print(f"Before filtering there are {dataset.count_boxes()} bboxes")
    for image_data in tqdm(dataset.images, desc="Filtering"):
        image_data.bboxes = filter_duplicate_bboxes(
            image_data.bboxes,
            iou_threshold=iou_threshold
        )

    print(f"After filtering there are {dataset.count_boxes()} bboxes")

    # Place for other preprocessing


def save_yolo_labels(dataset: Dataset, output_path: Path, split: str):
    output_path.mkdir(parents=True, exist_ok=True)
    out_labels = output_path / "labels" / split
    out_labels.mkdir(parents=True, exist_ok=True)

    for image_data in dataset.images:
        lines = []
        for bbox in image_data.bboxes:
            lines.append(
                f"{bbox.cat_id} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height}"
            )

        with open(out_labels / image_data.label_path.name, "w") as f:
            f.write("\n".join(lines))


def local_contrast_normalization(img: torch.Tensor, kernel_size: int = 3, eps: float = 1e-5):
    pad = kernel_size // 2
    mean = F.avg_pool2d(img, kernel_size, stride=1, padding=pad)
    sq_mean = F.avg_pool2d(img * img, kernel_size, stride=1, padding=pad)
    var = sq_mean - mean * mean
    std = torch.sqrt(torch.clamp(var, min=eps))

    centered = img - mean
    mask = std > 1.0
    normalized = torch.where(mask, centered / std, centered)
    return normalized


def local_response_normalization(img: torch.Tensor, size=5, alpha=1e-4, beta=0.75, k=2.0):
    return F.local_response_norm(img, size=size, alpha=alpha, beta=beta, k=k)


class DetectionDataset(TorchDataset):
    def __init__(self, dataset: Dataset, norm: str = "none", hflip_prob: float = 0.5):
        self.dataset = dataset
        self.norm = norm
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.dataset.images)

    def __getitem__(self, idx):
        data = self.dataset.images[idx]

        img = cv2.imread(str(data.image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)

        labels = torch.tensor([
            [bbox.cat_id, bbox.x_center, bbox.y_center, bbox.width, bbox.height]
            for bbox in data.bboxes
        ], dtype=torch.float32)

        if random.random() < self.hflip_prob:
            img = torch.flip(img, dims=[2])
            if labels.numel() > 0:
                labels[:, 1] = 1.0 - labels[:, 1]

        if self.norm == "lcn":
            img = local_contrast_normalization(img.unsqueeze(0)).squeeze(0)
        elif self.norm == "lrn":
            img = local_response_normalization(img.unsqueeze(0)).squeeze(0)

        return img, labels
    

def pipeline(yolo_dataset_path: Path, yolo_yaml_path: Path, output_path: Path, split: str = "train", norm_method="none"):
    yolo_imgs = yolo_dataset_path / "images" / split
    yolo_labels = yolo_dataset_path / "labels" / split

    dataset = read_yolo_dataset(yolo_imgs, yolo_labels, yolo_yaml_path)
    preprocess_dataset(dataset, iou_threshold=0.75)
    save_yolo_labels(dataset, output_path, split)


if __name__ == "__main__":
    yolo_dataset = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO")
    yolo_yaml_path = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO/yolo_dataset.yaml")
    output_path = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO_filtered")
    split = "train"

    dataset = pipeline(
        yolo_dataset,
        yolo_yaml_path,
        output_path,
        split,
    )

    train_dataset = DetectionDataset(dataset, norm="lrn", hflip_prob=0.5)