# Ultralytics installation
If typical `pip install ultralytics` does not work due to connection restrictions try installing with mirror:
```
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Dataset preprocessing
## Conversion to YOLO format
YOLO format:
- images
    - train
    - val
    - test
- labels
    - train
    - val
    - test

Labels are `txt` files named as corresponding images with line of values divided by " ":
``` 
category_number x_center y_center width height
```

Input correct paths in `coco2yolo.py` file. \
**Important**: I didn't wrote any script for managing images folders in YOLO format, so you need to do that **manually**! \

1. Make `images` folder with `train` and `test` subfolders with corresponding images split
2. Run `coco2yolo` conversion 
3. Run `train_val_split` (integrated in conversion) to make validation set.

## Bboxes filtering
Dataset description says:

> The dataset contains many duplicated bounding boxes for the same subject which we have not corrected. You will probably want to filter them by taking the IOU for classes that are 100% overlapping or it could affect your model performance (expecially in stoplight detection which seems to suffer from an especially severe case of duplicated bounding boxes)

In `preprocessing.py` there is a simple dataset structure for easier dataset processing. \
Currently only filtering added, other processings can be added there too.

Fuction `pipeline` is meant to read and process YOLO dataset.

---
CLI was not added in any of the scripts intentionally, because we can use the functions somewhere in the full pipeline script or something.