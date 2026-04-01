import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


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


def preprocess_image_folder(
    input_dir: Path,
    output_dir: Path,
    sigma: float = 50.0,
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8),
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("**/*.jpg"))

    for img_path in tqdm(image_paths, desc="Preprocessing images"):
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        processed = retinex_luminance(
            img,
            sigma=sigma,
            use_clahe=use_clahe,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size
        )

        # Keep same relative structure
        relative_path = img_path.relative_to(input_dir)
        save_path = output_dir / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), processed)


if __name__ == "__main__":
    images_dir = Path("./data/dm-2026-lab-3-object-detection/YOLO/images_orig")
    out_dir = Path("./data/dm-2026-lab-3-object-detection/YOLO/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocess_image_folder(
        input_dir=images_dir,
        output_dir=out_dir,
    )

