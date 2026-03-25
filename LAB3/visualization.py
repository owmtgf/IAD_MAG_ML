from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from utils import yaml_read, yolo_to_xyxy, generate_colors


def read_yolo_labels(label_path: Path):
    bboxes = []

    if not label_path.exists():
        return bboxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x, y, w, h = parts

            bboxes.append((
                int(class_id),
                float(x),
                float(y),
                float(w),
                float(h),
            ))

    return bboxes


def visualize_image(
    image_path: Path,
    label_path: Path,
    class_names: list[str],
    colors: list[tuple[int, int, int]],
    save_path: Path = None,
    border: int = 40,
):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # create padded background
    new_w = w + 2 * border
    new_h = h + 2 * border

    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(img, (border, border))

    draw = ImageDraw.Draw(canvas)

    # try to load font (fallback to default)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    bboxes = read_yolo_labels(label_path)

    for bbox in bboxes:
        class_id, x1, y1, x2, y2 = yolo_to_xyxy(bbox, w, h)

        # shift due to border
        x1 += border
        y1 += border
        x2 += border
        y2 += border

        color = colors[class_id]
        label = class_names[class_id]

        # draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # text background (top-left corner)
        text_bg = [x1, y1 - text_h - 4, x1 + text_w + 6, y1]
        draw.rectangle(text_bg, fill=color)

        # text
        draw.text(
            (x1 + 3, y1 - text_h - 2),
            label,
            fill=(255, 255, 255),
            font=font
        )

    if save_path:
        canvas.save(save_path)
    else:
        canvas.show()


def visualize_dataset_sample(
    images_dir: Path,
    labels_dir: Path,
    yaml_path: Path,
    output_dir: Path,
    max_images: int = 10,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = yaml_read(yaml_path)
    colors = generate_colors(len(class_names))

    image_paths = list(images_dir.glob("*"))

    for img_path in tqdm(image_paths[:max_images]):
        label_path = labels_dir / (img_path.stem + ".txt")
        save_path = output_dir / f"{img_path.stem}_viz.jpg"

        visualize_image(
            img_path,
            label_path,
            class_names,
            colors,
            save_path=save_path
        )


if __name__ == "__main__":
    images_dir = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO/images/train")
    labels_dir = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO_filtered/labels/train")
    yaml_path = Path("LAB3/data/dm-2026-lab-3-object-detection/YOLO/yolo_dataset.yaml")
    output_dir = Path("LAB3/data/dm-2026-lab-3-object-detection/vis")
    max_images = 20

    visualize_dataset_sample(images_dir, labels_dir, yaml_path, output_dir, max_images)