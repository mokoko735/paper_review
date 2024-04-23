import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2


class config():
    def __init__(self):
        self.batch_size = 16
        self.lr = 2e-5
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.lr_schedule = [
            [1, 1e-4],
            [2, 5e-5],
            [200, 2e-5],
            [1e+10, 1e-5],
        ]
        self.epochs = 160
        self.device='cuda'

        self.img_size = 448
        self.num_classes = 20

        self.iou_threshold = 0.5
        self.prob_threshold = 0.5

        self.S, self.B, self.C = 7, 2, 20

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = A.Compose([
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.1,
                rotate_limit=0, p=1.0, border_mode=cv2.BORDER_CONSTANT,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.ToGray(p=0.1),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5, label_fields=[]))

        self.test_transform = A.Compose([
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5, label_fields=[]))

        self.voc_classes = [
            'AEROPLANE',
            'BICYCLE',
            'BIRD',
            'BOAT',
            'BOTTLE',
            'BUS',
            'CAR',
            'CAT',
            'CHAIR',
            'COW',
            'DININGTABLE',
            'DOG',
            'HORSE',
            'MOTORBIKE',
            'PERSON',
            'POTTEDPLANT',
            'SHEEP',
            'SOFA',
            'TRAIN',
            'TVMONITOR'
        ]

        self.print_interval = 50
        self.save_interval = 10
        self.model_dir = '../checkpoints/experiment2'
        
        self.load_model = False
        self.load_model_path = os.path.join(self.model_dir, 'YOLOv1_best_model.pt')
        self.save_model_path = os.path.join(self.model_dir, 'YOLOv1_{epoch}epoch.pt')
        self.best_model_path = os.path.join(self.model_dir, 'YOLOv1_best_model.pt')
        self.load_pretrained_model = False
        self.pretrained_model_path = os.path.join(self.model_dir, 'pretrained_darknet.pt')

        self.dataset_dir = r'D:\AI\Dataset\VocDetection2'
        self.img_dir = os.path.join(self.dataset_dir, 'images')
        self.label_dir = os.path.join(self.dataset_dir, 'labels')
        self.train_csv = os.path.join(self.dataset_dir, 'train.csv')
        self.valid_csv = os.path.join(self.dataset_dir, 'valid.csv')

        self.save_csv_path = os.path.join(self.model_dir, 'YOLOv1_result.csv')










# experiment1, 2, 3에서 사용한 transform ------------------------------------------------------------------
# self.train_transform = A.Compose([
#     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
#     A.Blur(p=0.1),
#     A.CLAHE(p=0.1),
#     A.ToGray(p=0.1),
#     A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
#     A.LongestMaxSize(max_size=int(self.img_size * scale)),
#     A.PadIfNeeded(
#         min_height=int(self.img_size * scale),
#         min_width=int(self.img_size * scale),
#         border_mode=cv2.BORDER_CONSTANT,
#     ),
#     A.RandomCrop(width=self.img_size, height=self.img_size),
#     A.HorizontalFlip(p=0.5),
#     ToTensorV2(),
# ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[]))

# self.test_transform = A.Compose([
#     A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
#     A.LongestMaxSize(max_size=self.img_size),
#     A.PadIfNeeded(
#         min_height=self.img_size,
#         min_width=self.img_size,
#         border_mode=cv2.BORDER_CONSTANT,
#     ),
#     ToTensorV2(),
# ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[]))