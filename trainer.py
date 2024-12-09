from tqdm import tqdm
import torch
import numpy as np
from Const import device
import Util
from EarlyStopping import EarlyStopping
from FileManager import FileManager

class Trainer:
    def __init__(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def eval(self, eval_model):
        eval_model.eval()

        # 예측 테스트
        with torch.no_grad():
            pred = []
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                predicted = eval_model(data)
                pred = predicted.data.detach().cpu().numpy()

        return pred

    def compute_loss(self, model, data_loader, criterion):
        losses = []
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                loss = criterion(output, target)
                losses.append(loss.item())
        return np.mean(losses)

    def train(self, epochs, train_model, criterion, optimizer):
        early_stopping = EarlyStopping(patience=10, verbose=True)
        model = train_model
        train_loss_list, valid_loss_list, test_loss_list = [], [], []

        progress = tqdm(range(0, epochs))
        for epoch in progress:
            loss_list = []
            model.train()
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                output = model(data).squeeze()

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            train_loss_list.append(np.mean(loss_list))

            model.eval()
            with torch.no_grad():
                valid_loss = self.compute_loss(model, self.valid_loader, criterion)
                test_loss = self.compute_loss(model, self.test_loader, criterion)
                valid_loss_list.append(valid_loss)
                test_loss_list.append(test_loss)

            early_stopping(valid_loss, model)
            progress.set_postfix({'Train Loss': train_loss_list[-1], 'Valid Loss': valid_loss, 'Test Loss': test_loss})

            if early_stopping.early_stop:
                break

        return model, np.mean(valid_loss_list)
        # return np.mean(train_loss_list), np.mean(valid_loss_list), np.mean(test_loss_list)

