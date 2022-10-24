import torch
from torch.utils.data import Dataset
import os, json
import numpy as np

class AnimalDataset(Dataset):
    def __init__(self, split, path = 'data/animal'):
        super().__init__()
        self.NUM_LATENT = 77 * 768
        self.NUM_CLASS = 6
        self.split = split
        self.path = path
        with open(os.path.join(path, 'meta.json')) as f:
            self.meta = json.load(f)
        with open(os.path.join(path, f'{self.split}.json')) as f:
            self.files = json.load(f)
        

    def __getitem__(self, idx):
        latent = np.load(os.path.join(self.path, "npy", f"{self.files[idx]}.npy"))
        latent = latent[0].flatten()
        label = int(self.files[idx].split("_")[-1:][0])
        #label = torch.nn.functional.one_hot(torch.tensor([label]), num_classes=self.NUM_CLASS) #label.flatten().float(),
        return {
            "latent": torch.from_numpy(latent).float(),
            "label": label
        }

    
    def __len__(self):
        return len(self.files)