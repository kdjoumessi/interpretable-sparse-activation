import os
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

class MoorfieldsDataset(torch.utils.data.Dataset):
    def __init__(self, split: str="train", transform=None, target_transform=None):
        self.root_dir = "./dataset/"
        meta = pd.read_csv(os.path.join(self.root_dir, f"splits/{split}.csv"))
        self.images = meta.Image
        self.labels = meta.Label
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(np.sort(np.unique(self.labels.to_numpy())))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "good_quality/", self.images[idx])
        image = Image.open(image_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label