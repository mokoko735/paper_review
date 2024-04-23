import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2

from dataset import ImageNet
from model import DarkNet
from trainer import Trainer

class config():
    def __init__(self):
        self.batch_size = 32
        self.lr = 2e-5
        self.lr_schedule = [
            [30, 2e-5],
            [100, 1e-5],
            [1e+10, 1e-5],
        ]
        self.epochs = 50
        self.device='cuda'

        self.img_size = 448
        self.num_classes = 30

        scale = 1.1
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.train_transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.ToGray(p=0.1),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            A.LongestMaxSize(max_size=int(self.img_size * scale)),
            A.PadIfNeeded(
                min_height=int(self.img_size * scale),
                min_width=int(self.img_size * scale),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomCrop(width=self.img_size, height=self.img_size),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ])

        self.test_transform = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            ToTensorV2(),
        ])

        class_names = [
            'tench',
            'goldfish',
            'great white shark',
            'tiger shark',
            'hammerhead',
            'electric ray',
            'stingray',
            'cock',
            'hen',
            'ostrich',
            'brambling',
            'goldfinch',
            'house finch',
            'junco',
            'indigo bunting',
            'robin',
            'bulbul',
            'jay',
            'magpie',
            'chickadee',
            'water ouzel',
            'kite',
            'bald eagle',
            'vulture',
            'great grey owl',
            'European fire salamander',
            'common newt',
            'eft',
            'spotted salamander',
            'axolotl',
        ]

        self.print_interval = 50
        model_dir = './'
        self.load_model = True
        self.load_model_path = os.path.join(model_dir, 'YOLOv1_pretrained_30epoch.pt')
        self.save_model_path = os.path.join(model_dir, f'YOLOv1_pretrained_{self.epochs}epoch.pt')
        self.save_best_model = True
        self.best_model_path = os.path.join(model_dir, 'YOLOv1_pretrained_best_model.pt')

        dataset_dir = r'D:\AI\Dataset\ImageNet'
        self.img_dir = os.path.join(dataset_dir, 'images')
        self.train_csv = os.path.join(dataset_dir, 'train.csv')
        self.valid_csv = os.path.join(dataset_dir, 'valid.csv')
        # self.train_csv = os.path.join(dataset_dir, '120_sample.csv')
        # self.valid_csv = os.path.join(dataset_dir, '120_sample.csv')

        self.save_csv_path = './YOLOv1_pretrained_result.csv'

config = config()

train_dataset = ImageNet(
    config.train_csv,
    config.img_dir,
    config.train_transform,
)
valid_dataset = ImageNet(
    config.valid_csv,
    config.img_dir,
    config.test_transform,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
)

model = DarkNet(config.num_classes).to(config.device)
optimizer = optim.Adam(model.parameters(), config.lr)
crit = nn.CrossEntropyLoss(reduction="sum")

trainer = Trainer(model, optimizer, crit)
trainer.train(train_loader, valid_loader, config)