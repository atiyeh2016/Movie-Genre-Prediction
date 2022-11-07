#%% Importing 
from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

#%% Class Defenition
class FilmsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory of images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = pd.read_csv(csv_file)
        selected_columns = data[["image","genre"]]
        self.landmarks_frame = selected_columns.copy()
        self.landmarks_frame['genre'] = self.landmarks_frame['genre'].astype(dtype="category").cat.codes
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = np.array(self.landmarks_frame.iloc[idx, 1:].astype(int))[0]

        if self.transform:
            transformed_image = self.transform(image)
        
        sample = {'image': transformed_image, 'landmarks': label}

        return sample