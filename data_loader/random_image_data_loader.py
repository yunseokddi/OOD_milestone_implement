import torch
import numpy as np

from torch.utils.data import Dataset

class RandomImages(Dataset):
    def __init__(self, transform=None):
        self.data_file = np.load('datasets/unlabeled_datasets/300K_random_images/300K_random_images.npy')
        self.offset = 0
        self.transform = transform

    def __getitem__(self, index):
        index = (index + self.offset) % 300000
        img = self.data_file[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 300000