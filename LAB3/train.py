import torch
from pathlib import Path
from ultralytics import YOLO

def train(yolo_dataset_yaml: str):
    model = YOLO("/home/maria/Projects/IAD_MAG_ML/LAB3/data/model/yolo12x.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = model.train(
        data=yolo_dataset_yaml,
        epochs=10,
        imgsz=640,
        batch=8,
        device=device,
        workers=4,
        seed=42,
        augment=False,
        pretrained=True,
        verbose=True,
        project="/home/maria/Projects/IAD_MAG_ML/LAB3/runs",
        name="baseline",
    )

    return model, results


def validate(yolo_dataset_yaml: str, model):
    metrics = model.val(
        data=yolo_dataset_yaml,
        split="val",
        imgsz=640,
        batch=16,
        device=0,
    )

    return metrics


def run_pipeline(yolo_dataset_yaml: str):
    model, train_results = train(yolo_dataset_yaml)
    val_metrics = validate(model)

    print("\nFinal validation metrics:")
    print(val_metrics)

if __name__ == "__main__":
    yolo_dataset_yaml = "/home/maria/Projects/IAD_MAG_ML/LAB3/data/dm-2026-lab-3-object-detection/YOLO/yolo_dataset.yaml"
    run_pipeline(yolo_dataset_yaml)