from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2

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


def process_image(image_path: Path, method: str = "none"):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    if method == "lcn":
        img = local_contrast_normalization(img)
    elif method == "lrn":
        img = local_response_normalization(img)

    img = img.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    return img


def preprocess_dataset(dataset: Dataset, iou_threshold=0.9, norm_method="none", output_img_dir=None):
    # Bboxes filtering
    print(f"Before filtering there are {dataset.count_boxes()} bboxes")
    for image_data in tqdm(dataset.images, desc="Filtering"):
        image_data.bboxes = filter_duplicate_bboxes(
            image_data.bboxes,
            iou_threshold=iou_threshold
        )
    
    if norm_method != "none":
        img = process_image(image_data.image_path, method=norm_method)
        out_path = output_img_dir / image_data.image_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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


def pipeline(yolo_dataset_path: Path, yolo_yaml_path: Path, output_path: Path, split: str = "train", norm_method="none"):
    yolo_imgs = yolo_dataset_path / "images" / split
    yolo_labels = yolo_dataset_path / "labels" / split

    output_img_dir = output_path / "images" / split
    output_img_dir.mkdir(parents=True, exist_ok=True)

    dataset = read_yolo_dataset(yolo_imgs, yolo_labels, yolo_yaml_path)
    preprocess_dataset(dataset, iou_threshold=0.75, norm_method=norm_method, output_img_dir=output_img_dir)
    save_yolo_labels(dataset, output_path, split)


if __name__ == "__main__":
    yolo_dataset = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO")
    yolo_yaml_path = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO/yolo_dataset.yaml")
    output_path = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO_filtered")
    split = "train"

    pipeline(
        yolo_dataset,
        yolo_yaml_path,
        output_path,
        split,
        norm_method="lcn"
    )
