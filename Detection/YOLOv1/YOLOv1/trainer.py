import torch
from copy import deepcopy
import numpy as np
import pandas as pd
import time as t
import os
from tqdm import tqdm
from utils import get_bboxes, mAP


class Trainer():
    
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

    def _train(self, train_loader, config):
        self.model.train()

        train_idx = 0
        pred_bboxes, true_bboxes = [], []
        t_loss, t_loss_coord, t_loss_conf, t_loss_cls = 0, 0, 0, 0
        for batch_idx, (images, label_matrix) in enumerate(train_loader):
            images = images.to(config.device)
            label_matrix = label_matrix.to(config.device)

            pred = self.model(images)
            loss, loss_coord, loss_conf, loss_cls = self.crit(pred, label_matrix)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 누적 loss 계산
            t_loss += loss.item()
            t_loss_coord += loss_coord.item()
            t_loss_conf += loss_conf.item()
            t_loss_cls += loss_cls.item()

            # box 기록
            batch_pred_bboxes, batch_true_bboxes, train_idx = get_bboxes(train_idx,
                                                                         label_matrix,
                                                                         pred,
                                                                         config.S,
                                                                         config.iou_threshold,
                                                                         config.prob_threshold
                                                                    )
            pred_bboxes.extend(batch_pred_bboxes)
            true_bboxes.extend(batch_true_bboxes)

            # 결과 출력
            if batch_idx % config.print_interval == 0:
                print(f"[{batch_idx+1}/{len(train_loader)}] train loss: {loss.item():0.4f}")
        
        # 평균 loss 계산
        len_iter = len(train_loader)
        mean_loss = t_loss / len_iter
        mean_loss_coord = t_loss_coord / len_iter
        mean_loss_conf = t_loss_conf / len_iter
        mean_loss_cls = t_loss_cls / len_iter

        # mAP 계산
        average_precisions, mean_average_precision = mAP(pred_bboxes, true_bboxes, config.iou_threshold, config.num_classes)

        return [
            mean_loss,
            mean_loss_coord,
            mean_loss_conf,
            mean_loss_cls,
            mean_average_precision,
            average_precisions
        ]
    
    def _validate(self, valid_loader, config):
        self.model.eval()

        with torch.no_grad():
            train_idx = 0
            pred_bboxes, true_bboxes = [], []
            t_loss, t_loss_coord, t_loss_conf, t_loss_cls = 0, 0, 0, 0
            for batch_idx, (images, label_matrix) in enumerate(valid_loader):
                images = images.to(config.device)
                label_matrix = label_matrix.to(config.device)

                pred = self.model(images)
                loss, loss_coord, loss_conf, loss_cls = self.crit(pred, label_matrix)
                
                # 누적 loss 계산
                t_loss += loss.item()
                t_loss_coord += loss_coord.item()
                t_loss_conf += loss_conf.item()
                t_loss_cls += loss_cls.item()

                # box 기록
                batch_pred_bboxes, batch_true_bboxes, train_idx = get_bboxes(train_idx,
                                                                            label_matrix,
                                                                            pred,
                                                                            config.S,
                                                                            config.iou_threshold,
                                                                            config.prob_threshold
                                                                        )
                pred_bboxes.extend(batch_pred_bboxes)
                true_bboxes.extend(batch_true_bboxes)

                # 결과 출력
                if batch_idx % config.print_interval == 0:
                    print(f"[{batch_idx+1}/{len(valid_loader)}] valid loss: {loss.item():0.4f}")

            # 평균 loss 계산
            len_iter = len(valid_loader)
            mean_loss = t_loss / len_iter
            mean_loss_coord = t_loss_coord / len_iter
            mean_loss_conf = t_loss_conf / len_iter
            mean_loss_cls = t_loss_cls / len_iter

            # mAP 계산
            average_precisions, mean_average_precision = mAP(pred_bboxes, true_bboxes, config.iou_threshold, config.num_classes)

            return [
                mean_loss,
                mean_loss_coord,
                mean_loss_conf,
                mean_loss_cls,
                mean_average_precision,
                average_precisions
            ]
        
    def test(self, test_loader, config):
        self.load_model(config)
        
        self.model.eval()

        with torch.no_grad():
            train_idx = 0
            pred_bboxes, true_bboxes = [], []
            t_loss, t_loss_coord, t_loss_conf, t_loss_cls = 0, 0, 0, 0
            img_paths = []
            for batch_idx, (images, label_matrix, _img_paths) in enumerate(test_loader):
                images = images.to(config.device)
                label_matrix = label_matrix.to(config.device)

                pred = self.model(images)
                loss, loss_coord, loss_conf, loss_cls = self.crit(pred, label_matrix)
                
                # 누적 loss 계산
                t_loss += loss.item()
                t_loss_coord += loss_coord.item()
                t_loss_conf += loss_conf.item()
                t_loss_cls += loss_cls.item()

                # box 기록
                batch_pred_bboxes, batch_true_bboxes, train_idx = get_bboxes(train_idx,
                                                                            label_matrix,
                                                                            pred,
                                                                            config.S,
                                                                            config.iou_threshold,
                                                                            config.prob_threshold
                                                                        )
                pred_bboxes.extend(batch_pred_bboxes)
                true_bboxes.extend(batch_true_bboxes)

                # 이미지 주소 기록
                img_paths.extend(_img_paths)

                # 결과 출력
                if batch_idx % config.print_interval == 0:
                    print(f"[{batch_idx+1}/{len(test_loader)}] test loss: {loss.item():0.4f}")

            # 평균 loss 계산
            len_iter = len(test_loader)
            mean_loss = t_loss / len_iter
            mean_loss_coord = t_loss_coord / len_iter
            mean_loss_conf = t_loss_conf / len_iter
            mean_loss_cls = t_loss_cls / len_iter

            losses = [mean_loss, mean_loss_coord, mean_loss_conf, mean_loss_cls]

            return losses, pred_bboxes, true_bboxes, img_paths

    def train(self, train_loader, valid_loader, config):
        # 학습 초기값 설정
        self.lowest_loss = np.inf
        self.resume_epoch = 1

        if config.load_model:
            self.load_model(config)
            self.activate_requires_grad(self.model, state=True)

        if config.load_pretrained_model:
            self.load_pretrained_backbone(
                self.model.darknet,
                config.pretrained_model_path,
                config.device
            )
            self.activate_requires_grad(self.model.darknet, state=False)

        total_t = 0
        for epoch in range(self.resume_epoch, config.epochs+1):
            # epoch 시작        
            start_t = t.time()

            lr = self.get_lr(epoch, config.lr_schedule)
            self.update_lr(lr, self.optimizer)

            train_loss = self._train(train_loader, config)
            valid_loss = self._validate(valid_loader, config)

            if valid_loss[0] <= self.lowest_loss:
                self.lowest_loss = valid_loss[0]
                self.save_model(epoch, config.best_model_path)

            if epoch % config.save_interval == 0:
                self.save_model(epoch, config.save_model_path.format(epoch=epoch))

            # epoch 종료
            end_t = t.time()
            epoch_t = end_t - start_t
            total_t += epoch_t
            epoch_time = self._cal_train_time(epoch_t)
            total_time = self._cal_train_time(total_t)

            # 결과 출력
            print(
                f"[epoch][{epoch}/{config.epochs}] | "
                f"[batch][{epoch_time[0]}:{epoch_time[1]}:{epoch_time[2]}] | "
                f"[total][{total_time[0]}:{total_time[1]}:{total_time[2]}]\n"

                f"[train] total loss: {train_loss[0]:0.4f} | coordinate loss: {train_loss[1]:0.4f} | "
                f"confidence loss: {train_loss[2]:0.4f} | class loss: {train_loss[3]:0.4f} | mAP: {train_loss[4]:0.4f}\n"
                
                f"[validate] total loss: {valid_loss[0]:0.4f} | coordinate loss: {valid_loss[1]:0.4f} | "
                f"confidence loss: {valid_loss[2]:0.4f} | class loss: {valid_loss[3]:0.4f} | mAP: {valid_loss[4]:0.4f}\n"

                "\n---------------------------------------------------------------------------------------------------------------------------------\n"
            )

            # 결과 저장
            self.save_csv(epoch, train_loss, valid_loss, config)
        
        # 학습 종료 후 모델 저장
        self.save_model(epoch, config.save_model_path.format(epoch=epoch))
        
    def get_lr(self, epoch, lr_schedule):
        for limit_epoch, lr in lr_schedule:
            if epoch <= limit_epoch:
                return lr

    def update_lr(self, lr, optim):
        # print(f"\ncurrent learning rate: {lr}")
        optim.param_groups[0]['lr'] = lr

    def load_pretrained_backbone(self, model, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

    def activate_requires_grad(self, model, state=False):
        for p in model.parameters():
            p.requires_grad = state

    def load_model(self, config):
        print("\n\n\nnow loading.....")

        checkpoint = torch.load(config.load_model_path, map_location=config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.resume_epoch = checkpoint['resume_epoch']
        self.lowest_loss = checkpoint['lowest_loss']

        print(f"resume epoch: {self.resume_epoch} lowest loss: {self.lowest_loss}\n")
        
    def save_model(self, epoch, save_path):
        print("\nnow saving checkpoint.....\n")
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'resume_epoch': epoch + 1,
            'lowest_loss': self.lowest_loss,
        }, save_path)

    def save_csv(self, epoch, train_loss, valid_loss, config):
        if not os.path.isfile(config.save_csv_path):
            columns = ['train_loss', 'train_loss_coord', 'train_loss_conf', 'train_loss_cls', 'train_mAP']
            for i in range(len(train_loss[-1])):
                columns.append(f'train_AP{i}')

            columns.extend(['valid_loss', 'valid_loss_coord', 'valid_loss_conf', 'valid_loss_cls', 'valid_mAP'])
            for i in range(len(valid_loss[-1])):
                columns.append(f'valid_AP{i}')

            df = pd.DataFrame(columns=columns)
            df.to_csv(config.save_csv_path, index=False)
            
        df = pd.read_csv(config.save_csv_path)
        df.loc[epoch-1] = [*train_loss[:5], *train_loss[5], *valid_loss[:5], *valid_loss[5]]
        df.to_csv(config.save_csv_path, index=False)

    def _cal_train_time(self, train_time):
        train_hour, remainder = divmod(train_time, 3600)
        train_min, train_sec = divmod(remainder, 60)
        train_hour, train_min, train_sec = str(int(train_hour)).zfill(2), str(int(train_min)).zfill(2), str(int(train_sec)).zfill(2)

        return [train_hour, train_min, train_sec]