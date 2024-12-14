from tqdm import tqdm
import torch
import numpy as np
from Const import device
from EarlyStopping import EarlyStopping
from DataLoader import data_loader

class Trainer:
    def __init__(self, file_manager, train_loader, valid_loader, test_loader):
        self.file_manager = file_manager
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

                print("------")
                print("data ", data[0].squeeze())
                print("target", target[0])
                print("output", predicted[0])

                pred = predicted.data.detach().cpu().numpy()

        return pred

    def compute_loss(self, model, data_loader, criterion):
        model.eval()
        losses = []
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                loss = criterion(output, target)
                losses.append(loss.item())
        return np.mean(losses)

    def train_epoch(self, model, criterion, optimizer, loader):

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        model.train()
        loss_list = []
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).squeeze()
            #
            # print("------")
            # print("data ", data[0].squeeze())
            # print("target", target[0])
            # print("output", output[0])

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        scheduler.step()
        return np.mean(loss_list)

    def evaluate(self, model, loader, criterion):
        model.eval()
        with torch.no_grad():
            return self.compute_loss(model, loader, criterion)

    def train(self, epochs, train_model, train_criterion, train_optimizer):
        criterion = train_criterion
        optimizer = train_optimizer
        early_stopping = EarlyStopping(self.file_manager, patience=10, verbose=True)
        model = train_model.to(device)
        train_loss_list, valid_loss_list, test_loss_list = [], [], []

        progress = tqdm(range(0, epochs))
        for epoch in progress:
            train_loss = self.train_epoch(model, criterion, optimizer, self.train_loader)
            valid_loss = self.evaluate(model, self.valid_loader, criterion)
            test_loss = self.evaluate(model, self.test_loader, criterion)
            print(f"Epoch {epoch}, Loss: {train_loss}")
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)

            early_stopping(valid_loss, model)
            progress.set_postfix({
                'Train Loss': train_loss,
                'Valid Loss': valid_loss,
                'Test Loss': test_loss
            })

            if early_stopping.early_stop:
                break

        return np.mean(valid_loss_list)


def make_trainer(file_manager, values):
    _train_x, _train_y, _valid_x, _valid_y, _test_x, _test_y = values
    _train_loader, _valid_loader, _test_loader = data_loader(_train_x, _train_y, _valid_x, _valid_y, _test_x, _test_y)
    return Trainer(file_manager, _train_loader, _valid_loader, _test_loader)