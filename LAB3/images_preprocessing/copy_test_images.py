from pathlib import Path
from shutil import copy

images_idx = [
    190, 394, 587, 910, 988, 1017, 1093, 
    1304, 1363, 1465, 1551, 1600, 1694, 
    1888, 2112, 2123, 2259, 2588, 2726, 
    2978, 3276, 3340, 3536, 3677, 4121, 
    4541, 4792, 4999, 5133, 6624, 6690, 7080
]

root_im_dir = Path("./data/dm-2026-lab-3-object-detection/train_images/train_images")
out_dir = Path("./images_preprocessing/test_cases")
out_dir.mkdir(parents=True, exist_ok=True)

im_files = sorted(list(root_im_dir.glob("*.jpg")))
for idx in images_idx:
    copy(im_files[idx-1], out_dir)
