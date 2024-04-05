import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from trainer import Trainer
from row_data import RowData
from model_lstm import LTSF_LSTM
from Const import device
import os
from datetime import datetime

def show(path, real, result, input_window, output_window, show=False):
    fig = plt.figure(figsize=(20, 5))
    if result.shape[1] == 1:
        result = result[:, 0]
        plt.plot(range(input_window, len(result)+input_window), result, '--')
    else:
        for idx in range(0, len(result)):
            # if idx != 0 and idx % (config.output_window-1) != 0:
            #     continue
            start = input_window + idx
            end = input_window + idx + output_window
            plt.plot(range(start, end), result[idx], '--')

    plt.plot(range(0, len(real)), real, 'r,-')
    if show:
        plt.show()
    else:
        plt.savefig(path + '/result.png')

def path(epochs, input_window, output_window, hidden_size, types, scaler, learning_rate):
    scaler_str = 'empty'
    if type(scaler) == StandardScaler:
        scaler_str = 'Standard'

    dir = ('EP{}_'
           'IW{}_'
           'OW{}_'
           'HS{}_'
           'LR{}_'
           'SC{}'
           ).format(epochs, input_window, output_window, hidden_size, learning_rate, scaler_str)

    date_str = datetime.today().strftime("%Y%m%d")
    dir_path = './Model/{}/{}'.format(date_str, dir)
    os.makedirs(dir_path, exist_ok=True)
    f = open(dir_path + '/result.txt', 'w+')
    f.write('Epochs {}'.format(epochs))
    f.write('\nInputWindow {}'.format(input_window))
    f.write('\nOutputWindow {}'.format(output_window))
    f.write('\nHiddenSize {}'.format(hidden_size))
    f.write('\nLearningRate {}'.format(learning_rate))
    f.write('\nScaler {}'.format(scaler_str))
    f.write('\nTypes {}'.format(' '.join(types)))
    f.close()

    return dir_path

def main(epochs, input_window, output_window, hidden_size, types, scaler, learning_rate):
    row_data = RowData(
        input_window=input_window,
        output_window=output_window,
        types=types,
        scaler=scaler
    )

    dir_path = path(epochs, input_window, output_window, hidden_size, types, scaler, learning_rate)

    trainer = Trainer(path=dir_path, row_data=row_data)
    feature_size = len(types)
    lstm_model = LTSF_LSTM(output_window, feature_size, hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

    trainer.train(epochs, lstm_model, criterion, optimizer)
    result = trainer.eval(lstm_model)
    result = result[:, :, 0]
    result = scaler.inverse_transform(result)

    real = row_data.real_data

    show(dir_path, real, result, input_window, output_window)


if __name__ == "__main__":
    print(device)

    epochs = 1
    input_window = 24
    output_window = 2
    hidden_size = 128
    scaler = StandardScaler()
    learning_rate = 0.001
    types = ['Open', 'High', 'Low', 'Close', 'Volume']

    for idx in range(1, 6):
        main(
            epochs=epochs,
            input_window=input_window,
            output_window=output_window,
            hidden_size=hidden_size * idx,
            types=types,
            scaler=scaler,
            learning_rate=learning_rate
        )


