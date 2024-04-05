from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import numpy as np
from Const import device

class Trainer:
    def __init__(self, row_data, path):
        self.path = path
        train_x = row_data.train_x[:-300]
        train_y = row_data.train_y[:-300]
        valid_x = row_data.train_x[-300:]
        valid_y = row_data.train_y[-300:]
        test_x = row_data.test_x
        test_y = row_data.test_y

        train_data_set = TensorDataset(train_x, train_y)
        valid_data_set = TensorDataset(valid_x, valid_y)
        test_data_set = TensorDataset(test_x, test_y)
        self.train_loader = DataLoader(train_data_set, 64, shuffle=False)
        self.valid_loader = DataLoader(valid_data_set, valid_x.shape[0], shuffle=False)
        self.test_loader = DataLoader(test_data_set, test_x.shape[0], shuffle=False)

    def eval(self, train_model):
        model = train_model
        model.eval()

        # 예측 테스트
        with torch.no_grad():
            pred = []

            for data, target in self.test_loader:
                predicted = model(data.to(device))
                pred = predicted.data.detach().cpu().numpy()

        return pred

    def train(self, epochs, train_model, criterion, optimizer):
        model = train_model
        train_loss_list = []
        valid_loss_list = []
        test_loss_list = []

        max_loss = 999999999

        progress = tqdm(range(0, epochs))
        for epoch in progress:
            loss_list = []
            model.train()
            for data, target in self.train_loader:
                optimizer.zero_grad()
                output = model(data.float().to(device))
                output = output[:, :, 0]
                loss = criterion(output, target.float().to(device))
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            train_loss_list.append(np.mean(loss_list))

            model.eval()
            with torch.no_grad():
                for data, target in self.valid_loader:
                    output = model(data.float().to(device))
                    output = output[:, :, 0]
                    valid_loss = criterion(output, target.float().to(device))
                    valid_loss_list.append(valid_loss)

                for data, target in self.test_loader:
                    output = model(data.float().to(device))
                    output = output[:, :, 0]
                    test_loss = criterion(output, target.float().to(device))
                    test_loss_list.append(test_loss)

            if valid_loss < max_loss and epoch > (epochs / 2):
                # torch.save(train_model, self.path + '/model.pth')
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                        }, self.path + '/model.pth')
                max_loss = valid_loss
                print("valid_loss={:.6f}, test_los{:.6f}, Model Save".format(valid_loss, test_loss))
                best_epoch = epoch
                best_train_loss = np.mean(loss_list)
                best_valid_loss = np.mean(valid_loss.item())
                best_test_loss = np.mean(test_loss.item())
            print("epoch = {}, train_loss : {:.6f}, valid_loss : {:.6f}, test_loss : {:.6f}".format(epoch,
                                                                                                    np.mean(loss_list),
                                                                                                    valid_loss,
                                                                                                    test_loss))

        if 'best_epoch' in locals():
            f = open(self.path + '/result.txt', 'a+')
            f.write('\n\nbest_epoch: {}'.format(best_epoch))
            f.write('\nbest_train_loss: {}'.format(best_train_loss))
            f.write('\nbest_valid_loss: {}'.format(best_valid_loss))
            f.write('\nbest_test_loss: {}'.format(best_test_loss))
            f.close()
