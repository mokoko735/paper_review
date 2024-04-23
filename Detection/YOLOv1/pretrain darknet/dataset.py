import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image


class ImageNet(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        '''
        이미지 저장 경로: img_dir/label/file_name.JPEG
        '''
        label = self.annotations.iloc[index, 1]
        file_name = self.annotations.iloc[index, 0]
        file_name = os.path.join(str(label), file_name)
        image_path = os.path.join(self.img_dir, file_name)
        image = np.array(Image.open(image_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations['image']

        return image, label