import torch
from copy import deepcopy
import numpy as np
import pandas as pd
import time as t
import os

class Trainer():
    
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

    def _train(self, train_loader, config):
        self.model.train()

        t_loss = 0
        correct_cnt = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(config.device)
            labels = labels.to(config.device)

            pred = self.model(images)
            loss = self.crit(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 누적 loss 계산
            t_loss += loss.item()

            # 누적 정답수 계산
            labels = labels.detach().cpu()
            pred = pred.detach().cpu()
            correct_cnt += (labels.squeeze() == torch.argmax(pred, dim=-1)).sum()

            # 결과 출력
            if batch_idx % config.print_interval == 0:
                print(f"[{batch_idx+1}/{len(train_loader)}] train loss: {loss.item():0.4f}")
        
        # 평균 loss 및 accuracy 계산
        len_iter = len(train_loader)
        mean_loss = t_loss / len_iter
        accuracy = correct_cnt.numpy() / (len_iter * config.batch_size)

        return mean_loss, accuracy
    
    def _validate(self, valid_loader, config):
        self.model.eval()

        with torch.no_grad():
            t_loss = 0
            correct_cnt = 0
            for batch_idx, (images, labels) in enumerate(valid_loader):
                images = images.to(config.device)
                labels = labels.to(config.device)

                pred = self.model(images)
                loss = self.crit(pred, labels)
                
                # 누적 loss 계산
                t_loss += loss.item()

                # 누적 정답수 계산
                labels = labels.detach().cpu()
                pred = pred.detach().cpu()
                correct_cnt += (labels.squeeze() == torch.argmax(pred, dim=-1)).sum()

                # 결과 출력
                if batch_idx % config.print_interval == 0:
                    print(f"[{batch_idx+1}/{len(valid_loader)}] valid loss: {loss.item():0.4f}")

            # 평균 loss 및 accuracy 계산
            len_iter = len(valid_loader)
            mean_loss = t_loss / len_iter
            accuracy = correct_cnt.numpy() / (len_iter * config.batch_size)

            return mean_loss, accuracy

    def train(self, train_loader, valid_loader, config):
        # 학습 초기값 설정
        self.lowest_loss = np.inf
        self.resume_epoch = 1
        if config.load_model:
            self.load_model(config)

        total_t = 0
        for epoch in range(self.resume_epoch, config.epochs+1):
            # epoch 시작        
            start_t = t.time()

            lr = self.get_lr(epoch, config.lr_schedule)
            self.update_lr(lr, self.optimizer)

            train_loss, train_accuracy = self._train(train_loader, config)
            valid_loss, valid_accuracy = self._validate(valid_loader, config)

            if config.save_best_model:
                if valid_loss <= self.lowest_loss:
                    self.lowest_loss = valid_loss
                    self.save_model(epoch, config.best_model_path)

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

                f"[train] loss: {train_loss:0.4f} | accuracy: {train_accuracy:0.4f}\n"
                f"[validate] loss: {valid_loss:0.4f} | accuracy: {valid_accuracy:0.4f}\n"

                "\n---------------------------------------------------------------------------------------------------------------------------------\n"
            )

            # 결과 저장
            self.save_csv(epoch, train_loss, train_accuracy, valid_loss, valid_accuracy, config)

        # 모델 저장
        self.save_model(epoch, config.save_model_path)

    def test(self, test_loader, config):
        self.model.eval()

        with torch.no_grad():
            t_loss = 0
            correct_cnt = 0
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(config.device)
                labels = labels.to(config.device)

                pred = self.model(images)
                loss = self.crit(pred, labels)
                
                # 누적 loss 계산
                t_loss += loss.item()

                # 누적 정답수 계산
                labels = labels.detach().cpu()
                pred = pred.detach().cpu()
                correct_cnt += (labels.squeeze() == torch.argmax(pred, dim=-1)).sum()

                # 결과 출력
                if batch_idx % config.print_interval == 0:
                    print(f"[{batch_idx+1}/{len(test_loader)}] valid loss: {loss.item():0.4f}")

            # 평균 loss 및 accuracy 계산
            len_iter = len(test_loader)
            mean_loss = t_loss / len_iter
            accuracy = correct_cnt.numpy() / (len_iter * config.batch_size)

            return mean_loss, accuracy
        
    def get_lr(self, epoch, lr_schedule):
        for limit_epoch, lr in lr_schedule:
            if epoch <= limit_epoch:
                return lr

    def update_lr(self, lr, optim):
        # print(f"\ncurrent learning rate: {lr}")
        optim.param_groups[0]['lr'] = lr

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

    def save_csv(self, epoch, train_loss, train_accuracy, valid_loss, valid_accuracy, config):
        if not os.path.isfile(config.save_csv_path):
            columns = ['train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy']
            df = pd.DataFrame(columns=columns)
            df.to_csv(config.save_csv_path, index=False)
            
        df = pd.read_csv(config.save_csv_path)
        df.loc[epoch-1] = [train_loss, train_accuracy, valid_loss, valid_accuracy]
        df.to_csv(config.save_csv_path, index=False)

    def _cal_train_time(self, train_time):
        train_hour, remainder = divmod(train_time, 3600)
        train_min, train_sec = divmod(remainder, 60)
        train_hour, train_min, train_sec = str(int(train_hour)).zfill(2), str(int(train_min)).zfill(2), str(int(train_sec)).zfill(2)

        return [train_hour, train_min, train_sec]