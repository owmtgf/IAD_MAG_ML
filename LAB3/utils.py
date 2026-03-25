import yaml
from yaml import Loader
from pathlib import Path


def yaml_read(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        raise ValueError(f"YAML file {yaml_path} does not exist!")
    
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=Loader)
    
    names = data.get("names")
    if not names:
        raise ValueError(f"No 'names' field in yaml {yaml_path}")
    
    return names


def compute_iou(xyxy1: tuple, xyxy2: tuple) -> float:
    x1_1, y1_1, x2_1, y2_1 = xyxy1
    x1_2, y1_2, x2_2, y2_2 = xyxy2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union = area1 + area2 - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def generate_colors(num_classes: int) -> list[tuple[int, int, int]]:
    colors = []

    for i in range(num_classes):
        # simple deterministic palette
        r = (37 * i) % 255
        g = (17 * i + 100) % 255
        b = (29 * i + 200) % 255

        # ensure it's not too bright (so white text is visible)
        r = int(r * 0.7)
        g = int(g * 0.7)
        b = int(b * 0.7)

        colors.append((r, g, b))

    return colors


def yolo_to_xyxy(bbox, img_w, img_h):
    class_id, x, y, w, h = bbox

    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h

    return class_id, int(x1), int(y1), int(x2), int(y2)