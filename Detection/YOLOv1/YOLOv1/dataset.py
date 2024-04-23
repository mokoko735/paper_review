import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from config import config
from utils import draw, set_color, cell_to_xywh


class VOCDataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            label_dir,
            img_size,
            S=7, B=2, C=20,
            transform=None,
            return_img_path=False
        ):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.S, self.B, self.C = S, B, C
        self.transform = transform
        self.return_img_path = return_img_path

    def __len__(self):
        return len(self.annotations)
    
    def _data_encode(self, class_labels, boxes):
        '''
        bboxes to label_matrix
        parameters:
            bboxes(list) : [class_label, x_center, y_center, width, height]
            label_matrix(tensor) : [S, S, C + 5 * B]
        '''
        label_matrix = torch.zeros(self.S, self.S, self.C + 5 * self.B)
        for class_label, box in zip(class_labels, boxes):
            x, y, w, h = box

            j = int(self.S * x)
            i = int(self.S * y)
            x_cell = (self.S * x) - j
            y_cell = (self.S * y) - i
            w_cell = self.S * w
            h_cell = self.S * h

            # convert from midpoint to matrix format
            if label_matrix[i, j, 20] == 0:
                # object score
                label_matrix[i, j, 20] = 1

                # box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates

                # onehot encoding for class label
                label_matrix[i, j, class_label] = 1

        return label_matrix
        
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        class_labels = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = list(label.split(','))
                boxes.append([float(x), float(y), float(w), float(h)])
                class_labels.append(int(class_label))

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations['image']
            boxes = augmentations['bboxes']

        label_matrix = self._data_encode(class_labels, boxes)
        
        if self.return_img_path:
            data = [image, label_matrix, img_path]
        else:
            data = [image, label_matrix]

        return data


if __name__ == "__main__":
    config = config()
    
    dataset = VOCDataset(
        config.train_csv,
        config.img_dir,
        config.label_dir,
        config.img_size,
        config.S, config.B, config.C,
        config.train_transform,
        return_img_path=True,
    )

    # # 입출력 텐서 사이즈 확인 코드
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    # for images, label_matrix, img_paths in dataloader:
    #     print(images.shape)
    #     print(label_matrix.shape)
    #     print(img_paths)
    #     break

    # 출력 이미지 확인 코드
    class_names = config.voc_classes
    colors = set_color(len(class_names))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for images, labels, img_paths in dataloader:
        for i in range(len(images)):
            _bboxes = cell_to_xywh(labels[i].unsqueeze(0), config.S, format='target')
            _bboxes = _bboxes.squeeze()
            bboxes = []
            for bbox in _bboxes:
                if bbox[1] == 1:
                    bboxes.append(bbox)
                    
            draw(images[i], bboxes, class_names, colors, config.mean, config.std)