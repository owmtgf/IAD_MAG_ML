from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import csv
import json
import pandas as pd


# ---------- CONFIG ----------
# MODEL_PATH = Path("./runs/detect/runs/baseline2/weights/best.pt")
MODEL_PATH = Path("./kaggle/best.pt")
IMAGE_DIR = Path("./data/dm-2026-lab-3-object-detection/YOLO/images/test")
JSON_PATH = Path("./data/dm-2026-lab-3-object-detection/test_file_names.json")
SAMPLE_CSV = Path("./data/dm-2026-lab-3-object-detection/submission.csv")
OUTPUT_CSV = Path("submission.csv")
BATCH_SIZE = 8


# ---------- LOAD JSON MAPPING ----------
def load_image_mapping(json_path):
    with open(json_path) as f:
        data = json.load(f)

    mapping = {}
    for img in data["images"]:
        mapping[img["file_name"]] = img["id"]

    return mapping


# ---------- HELPERS ----------
def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def batch_iter(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_prediction_map(model, image_paths, id_map, batch_size=8, max_per_image=10):
    pred_map = {}

    for batch_paths in tqdm(list(batch_iter(image_paths, batch_size)), desc="Inference"):
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=640,
            conf=0.01,
            iou=0.7,
            device=0,
            verbose=False
        )

        for r, img_path in zip(results, batch_paths):
            file_name = img_path.name
            image_id = id_map[file_name]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = list(zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy()
            ))

            # boxes = sorted(boxes, key=lambda x: x[2], reverse=True)
            # boxes = boxes[:max_per_image]

            preds = []
            for box, cls, conf in boxes:
                preds.append({
                    "category_id": int(cls),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(conf)
                })

            pred_map[image_id] = preds

    return pred_map


def build_submission_from_sample(sample_csv, output_csv, pred_map):
    df = pd.read_csv(sample_csv)
    num_unique_images = df['image_id'].nunique()
    print(f"Number of unique images: {num_unique_images}")


    # group rows by image_id
    grouped = df.groupby("image_id")

    new_rows = []

    for image_id, group in grouped:
        preds = pred_map.get(image_id, [])

        for i, (_, row) in enumerate(group.iterrows()):
            if i < len(preds):
                pred = preds[i]

                row["category_id"] = pred["category_id"]
                row["bbox"] = str(pred["bbox"])
                row["score"] = pred["score"]
            else:
                row["category_id"] = -1
                row["bbox"] = "[0, 0, 0, 0]"
                row["score"] = 0.0

            new_rows.append(row)

    df = pd.DataFrame(new_rows)
    df.to_csv(output_csv, index=False)


def run_pipeline():
    model = YOLO(MODEL_PATH)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    id_map = load_image_mapping(JSON_PATH)

    pred_map = build_prediction_map(model, image_paths, id_map, BATCH_SIZE)

    build_submission_from_sample(
        sample_csv=SAMPLE_CSV,
        output_csv=OUTPUT_CSV,
        pred_map=pred_map
    )

    print(f"✅ Submission saved to: {OUTPUT_CSV.resolve()}")


# ---------- RUN ----------
if __name__ == "__main__":
    run_pipeline()
    