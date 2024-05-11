import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from trainer import Trainer
from RowData import RowData, sliding
from model_lstm import LTSF_LSTM
from Const import device
import os
from datetime import datetime
import numpy as np
import FinanceDataReader as fdr
from StockData import StockData, StockDataGenerator
from torch.utils.data import TensorDataset, DataLoader

def show(path, real, result, input_window, output_window, show=False):
    fig = plt.figure(figsize=(20, 5))
    start = 300
    count = 0
    if result.shape[1] == 1:
        # Output window == 1
        result = result[:, 0]
        for i in range(start, len(result)):
            real_prev_sign = (real[i+input_window] - real[i+input_window - 1]) >= 0
            pred_prev_sign = (result[i] - real[i+input_window - 1]) >= 0
            if i == start:
                plt.plot(i + input_window, result[i], 'k.')
            elif real_prev_sign == pred_prev_sign:
                plt.plot(i + input_window, result[i], 'b.')
                count += 1
            else:
                plt.plot(i + input_window, result[i], 'r.')

        # plt.plot(range(input_window, len(result) + input_window), result, '.')
    else:
        # Output window > 1
        for idx in range(start, len(result)):
            # if idx != 0 and idx % (config.output_window-1) != 0:
            #     continue
            start = input_window + idx
            end = input_window + idx + output_window
            plt.plot(range(start, end), result[idx], '.-')
    plt.title('Count {}/{}, {}'.format(count, len(result) - start, count/(len(result)-start)))
    plt.plot(range(start, len(real)), real[start:], 'k.-')
    if show:
        plt.show()
    else:
        plt.savefig(path + '/result.png')


def path(ep, iw, ow, hs, lr):
    directory = 'EP{}_IW{}_OW{}_HS{}_LR{}'.format(ep, iw, ow, hs, lr)
    date_str = datetime.today().strftime("%Y%m%d")
    dir_path = './Model/{}/{}'.format(date_str, directory)
    return dir_path


def make_file(dir_path, file_path, ep, iw, ow, hs, lr):
    os.makedirs(dir_path, exist_ok=True)
    f = open(file_path, 'w+')
    f.write('Epochs {}'.format(ep))
    f.write('\nInputWindow {}'.format(iw))
    f.write('\nOutputWindow {}'.format(ow))
    f.write('\nHiddenSize {}'.format(hs))
    f.write('\nLearningRate {}'.format(lr))
    f.close()


def main(data, epochs, input_window, output_window, hidden_size, scaler, learning_rate):
    dir_path = path(epochs, input_window, output_window, hidden_size, learning_rate)
    file_path = dir_path + '/result.txt'
    make_file(dir_path, file_path, epochs, input_window, output_window, hidden_size, learning_rate)

    train_x, train_y = sliding(data.train_target, data.train_data, input_window, output_window)
    valid_x, valid_y = sliding(data.valid_target, data.valid_data, input_window, output_window)
    test_x, test_y = sliding(data.test_target, data.test_data, input_window, output_window)

    train_data_set = TensorDataset(train_x, train_y)
    valid_data_set = TensorDataset(valid_x, valid_y)
    test_data_set = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_data_set, 64, shuffle=False)
    valid_loader = DataLoader(valid_data_set, valid_x.shape[0], shuffle=False)
    test_loader = DataLoader(test_data_set, test_x.shape[0], shuffle=False)

    trainer = Trainer(dir_path, train_loader, valid_loader, test_loader)
    lstm_model = LTSF_LSTM(output_window, train_x.shape[2], hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

    trainer.train(epochs, lstm_model, criterion, optimizer, scaler)
    result = trainer.eval(lstm_model)
    result = result[:, :, 0]
    result = scaler.inverse_transform(result)

    real = data.real

    show(dir_path, real, result, input_window, output_window)


if __name__ == "__main__":
    print(device)
    param_epochs = 200
    param_input_window = 20
    param_output_window = 1
    param_hidden_size = 128
    param_learning_rate = 0.001

    generator = StockDataGenerator()
    stock_data = generator.stock_data
    scaler = generator.scaler

    for idx in range(1, 5):
        for idx2 in range(1, 5):
            main(
                data=stock_data,
                epochs=param_epochs,
                input_window=param_input_window * idx2,
                output_window=param_output_window,
                hidden_size=param_hidden_size * idx,
                scaler=scaler,
                learning_rate=param_learning_rate
            )
