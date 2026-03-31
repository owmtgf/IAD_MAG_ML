import torch
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.dataset

import cv2
import numpy as np

from dataset import DetectionDataset


def retinex_luminance(
    img_bgr: np.ndarray,
    sigma: float = 50.0,
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8)
):
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # --- Step 1: Retinex on luminance ---
    y_float = y.astype(np.float32) + 1.0  # avoid log(0)

    blur = cv2.GaussianBlur(y_float, (0, 0), sigma)
    retinex = np.log(y_float) - np.log(blur + 1e-5)

    # Normalize back to [0, 255]
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    y = retinex.astype(np.uint8)

    # --- Step 2: Optional CLAHE (recommended) ---
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        y = clahe.apply(y)

    # Merge back
    ycrcb = cv2.merge([y, cr, cb])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return out

class RetinexPreprocess:
    def __init__(self, sigma=50):
        self.sigma = sigma

    def __call__(self, img):

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(img)}")

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = retinex_luminance(img_bgr, sigma=self.sigma)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img
    

class PreprocessWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess_fn=None):
        self.dataset = dataset
        self.preprocess_fn = preprocess_fn

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        if isinstance(data, dict) and "img" in data:
            img = data["img"]

            if self.preprocess_fn is not None:
                img = self.preprocess_fn(img)

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            if img.ndim == 3 and img.shape[2] == 3:
                img = img.permute(2, 0, 1)

            img = img.float() / 255.0

            data["img"] = img

        return data
    
class MyTrainer(DetectionTrainer):
    def __init__(self, *args, preprocess_fn=None, **kwargs):
        self.preprocess_fn = preprocess_fn
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)

        if mode == "train":
            return PreprocessWrapper(dataset, self.preprocess_fn)

        return dataset


def train(yolo_dataset_yaml: str, name: str = "baseline"):
    trainer = MyTrainer(
        overrides=dict(
            model="yolo12n.pt",
            data=yolo_dataset_yaml,
            epochs=3,
            imgsz=640,
            batch=4,
            device=0,
            workers=2,
            augment=True,
            pretrained=True,
            verbose=True,
            project="runs",
            name=name,
        ),
        preprocess_fn=RetinexPreprocess(sigma=50),
    )

    trainer.train()

    trained_model = YOLO(trainer.best)

    return trained_model, trainer


def validate(yolo_dataset_yaml: str, model):
    metrics = model.val(
        data=yolo_dataset_yaml,
        split="val",
        imgsz=640,
        batch=16,
        device=0,
    )
    return metrics


def run_pipeline(yolo_dataset_yaml: str, name: str = "baseline"):
    model, trainer = train(yolo_dataset_yaml, name)

    val_metrics = validate(yolo_dataset_yaml, model)

    print("\nFinal validation metrics:")
    print(val_metrics)

    return model

if __name__ == "__main__":
    yolo_dataset_yaml = "./data/dm-2026-lab-3-object-detection/YOLO/yolo_dataset.yaml"
    run_pipeline(yolo_dataset_yaml)
    