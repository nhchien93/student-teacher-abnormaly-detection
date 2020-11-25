import glob

import torch
from torch.utils.data import Dataset

import utils

class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_data =  "grid/train/good", transform = None):
        self.path_data = path_data
        self.transform = transform
        self.list_path = glob.glob(self.path_data + "/*.png")

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = utils.load_image_(self.list_path[idx])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}

        return sample