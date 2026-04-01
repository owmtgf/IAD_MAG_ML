from pathlib import Path
import json
import pandas as pd
import cv2
import ast
from tqdm import tqdm

# ---------- CONFIG ----------
CSV_PATH = Path("submission.csv")
JSON_PATH = Path("./data/dm-2026-lab-3-object-detection/test_file_names.json")
IMAGE_DIR = Path("./data/dm-2026-lab-3-object-detection/test_images/test_images")
OUTPUT_DIR = Path("submission_visualizations")

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------- LOAD IMAGE MAPPING ----------
def load_image_mapping(json_path):
    with open(json_path) as f:
        data = json.load(f)

    return {img["id"]: img["file_name"] for img in data["images"]}


# ---------- DRAW ----------
def draw_boxes(image, rows):
    for _, row in rows.iterrows():
        bbox = ast.literal_eval(row["bbox"])
        x, y, w, h = map(int, bbox)

        x2, y2 = x + w, y + h

        label = f"{row['category_id']}:{row['score']:.2f}"

        # draw box
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

        # draw label
        cv2.putText(
            image,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return image


# ---------- MAIN ----------
def visualize():
    df = pd.read_csv(CSV_PATH)
    id_map = load_image_mapping(JSON_PATH)

    for image_id, group in tqdm(df.groupby("image_id")):
        file_name = id_map.get(image_id)

        if file_name is None:
            continue

        img_path = IMAGE_DIR / file_name
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))

        image = draw_boxes(image, group)

        out_path = OUTPUT_DIR / f"{image_id}.jpg"
        cv2.imwrite(str(out_path), image)

    print(f"✅ Visualizations saved to: {OUTPUT_DIR.resolve()}")


# ---------- RUN ----------
if __name__ == "__main__":
    visualize()