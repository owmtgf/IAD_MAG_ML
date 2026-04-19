import torch
from ultralytics import YOLO

def train(yolo_dataset_yaml: str, name: str = 'baseline', **kwargs):
    model = YOLO("data/model/yolo12m.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = model.train(
        data=yolo_dataset_yaml,
        epochs=1,
        imgsz=640,
        batch=16,
        device=device,
        workers=1,
        seed=42,
        pretrained=True,
        verbose=True,
        project="runs",
        name=name,
<<<<<<< Updated upstream
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,  # mosaic augmentation
        mixup=0.0,
        erasing=0.0,
        crop_fraction=0.0,
        auto_augment=None,
=======
        # augment=False,
        # hsv_h=0.0,
        # hsv_s=0.0,
        # hsv_v=0.0,
        # degrees=0.0,
        # translate=0.0,
        # scale=0.0,
        # shear=0.0,
        # perspective=0.0,
        # flipud=0.0,
        # fliplr=0.0,
        # mosaic=0.0,
        # mixup=0.0,
        # erasing=0.0,
        # crop_fraction=0.0,
        # auto_augment=None,
        # cls=2.0,
>>>>>>> Stashed changes
        **kwargs,
    )
    return model, results


def validate(yolo_dataset_yaml: str, model):
    metrics = model.val(
        data=yolo_dataset_yaml,
        split="val",
        imgsz=640,
        batch=32,
        device=0,
    )
    return metrics


def run_pipeline(yolo_dataset_yaml: str, name: str = 'baseline', **kwargs):
    model, train_results = train(yolo_dataset_yaml, name, **kwargs)
    val_metrics = validate(yolo_dataset_yaml, model)

    # print("\nFinal validation metrics:")
    # print(val_metrics)

if __name__ == "__main__":
    yolo_dataset_yaml = "data/dm-2026-lab-3-object-detection/YOLO/yolo_dataset.yaml"
<<<<<<< Updated upstream
    run_pipeline(yolo_dataset_yaml, name="baseline")
=======
    run_pipeline(yolo_dataset_yaml, name="v12m_b8_base_labels")
>>>>>>> Stashed changes
