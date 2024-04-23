import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import cv2

from dataset import VOCDataset
from model import YOLOv1
from loss import YoloLoss1, YoloLoss2
from trainer import Trainer
from config import config

config = config()

config.batch_size = 16
config.lr = 2e-5
# config.momentum = 0.9
# config.weight_decay = 0.0005
config.lr_schedule = [
    [10, 5e-5],
    [200, 2e-5],
    [1e+10, 1e-5],
]
config.epochs = 50
config.device='cuda'

config.print_interval = 50
config.save_interval = 10
config.model_dir = '../checkpoints/experiment4'

config.load_model = True
config.load_model_path = os.path.join(config.model_dir, 'YOLOv1_10epoch.pt')
config.save_model_path = os.path.join(config.model_dir, 'YOLOv1_{epoch}epoch.pt')
config.best_model_path = os.path.join(config.model_dir, 'YOLOv1_best_model.pt')
config.load_pretrained_model = False
config.pretrained_model_path = os.path.join(config.model_dir, 'pretrained_darknet.pt')

config.dataset_dir = r'D:\AI\Dataset\VocDetection2'
config.img_dir = os.path.join(config.dataset_dir, 'images')
config.label_dir = os.path.join(config.dataset_dir, 'labels')
config.train_csv = os.path.join(config.dataset_dir, 'train.csv')
config.valid_csv = os.path.join(config.dataset_dir, 'valid.csv')
# config.train_csv = os.path.join(config.dataset_dir, '16_sample_train.csv')
# config.valid_csv = os.path.join(config.dataset_dir, '16_sample_valid.csv')

config.save_csv_path = os.path.join(config.model_dir, 'YOLOv1_result.csv')

train_dataset = VOCDataset(
    config.train_csv,
    config.img_dir,
    config.label_dir,
    config.img_size,
    config.S, config.B, config.C,
    config.train_transform,
)
valid_dataset = VOCDataset(
    config.valid_csv,
    config.img_dir,
    config.label_dir,
    config.img_size,
    config.S, config.B, config.C,
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

model = YOLOv1(config.S, config.B, config.C).to(config.device)
optimizer = optim.Adam(model.parameters(), config.lr)
# optimizer = optim.SGD(
#     model.parameters(),
#     lr=config.lr,
#     momentum=config.momentum,
#     weight_decay=config.weight_decay,
# )
crit = YoloLoss2(config.S, config.B, config.C)

trainer = Trainer(model, optimizer, crit)
trainer.train(train_loader, valid_loader, config)