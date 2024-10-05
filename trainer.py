from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import numpy as np
from Const import device
import Util

class Trainer:
    def __init__(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def eval(self, eval_model):
        checkpoint = torch.load(Util.path()+'/model.pth')
        eval_model.load_state_dict(checkpoint['state_dict'])

        eval_model.eval()

        # 예측 테스트
        with torch.no_grad():
            pred = []

            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)

                predicted = eval_model(data)
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
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                output = model(data)
                output = output[:, 0, 0]

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            train_loss_list.append(np.mean(loss_list))

            model.eval()
            with torch.no_grad():
                for data, target in self.valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    output = output[:, 0:, 0]

                    model_loss = criterion(output, target)
                    valid_loss = model_loss
                    valid_loss_list.append(valid_loss.item())

                for data, target in self.test_loader:
                    data, target = data.to(device), target.to(device)

                    output = model(data)
                    output = output[:, 0:, 0]

                    model_loss = criterion(output, target)
                    test_loss = model_loss
                    test_loss_list.append(test_loss.item())

            if valid_loss < max_loss and epoch > (epochs / 2):
                # torch.save(train_model, self.path + '/model.pth')
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                        }, Util.path() + '/model.pth')
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
            f = open(Util.file_path(), 'a+')
            f.write('\n\nbest_epoch: {}'.format(best_epoch))
            f.write('\nbest_train_loss: {}'.format(best_train_loss))
            f.write('\nbest_valid_loss: {}'.format(best_valid_loss))
            f.write('\nbest_test_loss: {}'.format(best_test_loss))
            f.close()

        return train_loss_list, valid_loss_list, test_loss_list
