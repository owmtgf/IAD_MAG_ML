import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from preprocessing import local_contrast_normalization, local_response_normalization

np.random.seed(42)


class DetectionDataset(Dataset):
    def __init__(self, dataset: Dataset, norm: str = "none", hflip_prob: float = 0.5):
        self.dataset = dataset
        self.norm = norm
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.dataset.images)

    def __getitem__(self, idx):
        data = self.dataset.images[idx]

        img = cv2.imread(str(data.image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)

        labels = torch.tensor([
            [bbox.cat_id, bbox.x_center, bbox.y_center, bbox.width, bbox.height]
            for bbox in data.bboxes
        ], dtype=torch.float32)

        if np.random.random() < self.hflip_prob:
            img = torch.flip(img, dims=[2])
            if labels.numel() > 0:
                labels[:, 1] = 1.0 - labels[:, 1]

        if self.norm == "lcn":
            img = local_contrast_normalization(img.unsqueeze(0)).squeeze(0)
        elif self.norm == "lrn":
            img = local_response_normalization(img.unsqueeze(0)).squeeze(0)

        return img, labels
    