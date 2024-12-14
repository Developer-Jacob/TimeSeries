import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import Const
import Parser


def draw_test(data1, data2):
    fig = plt.figure(figsize=(20, 5))

    d = data1[:500]


    plt.plot(range(0, len(d)), d, 'b.-')
    plt.show()
    if data2:
        plt.plot(range(0, len(data2)), data2, 'r.-')
        plt.show()

def draw_data_target(train, valid, test):
    fig = plt.figure(figsize=(20, 5))

    plt.plot(range(0, len(train)), train, 'b.-')
    plt.plot(range(len(train), len(train)+len(valid)), valid, 'r.-')
    plt.plot(range(len(train)+len(valid), len(train)+len(valid)+len(test)), test, 'y.-')
    plt.show()

def draw_result(real, result, path, section=100):
    fig = plt.figure(figsize=(20, 5))
    start = len(real) - section
    if start < 0:
        start = 0
    end = len(real)
    plt.plot(range(start, end), result[start:end], 'r.-')
    plt.plot(range(start, end), real[start:end], 'b.-')
    plt.savefig(path)


def draw_variance(diffed, pred, path, section=100):
    fig = plt.figure(figsize=(20, 5))

    count = section
    if len(pred) < section:
        count = len(pred)

    categories = list(range(0, count))  # range를 리스트로 변환
    x = np.arange(len(categories))  # X축 위치 인덱스

    # 두 개의 데이터에 대해 막대 그래프 생성
    plt.bar(x - 0.2, diffed[-count:], width=0.4, label='Real', color='blue')
    plt.bar(x + 0.2, pred.reshape(-1)[-count:], width=0.4, label='Pred', color='red')

    plt.title('Grouped Bar Chart')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(path)


def print_result(path, real, diffed, pred, convert_pred):
    f = open(path, 'a+')
    f.write("\n\n값 비교")
    f.write('\n real: {}'.format(real[-10:]))
    f.write('\n\n convert pred: {}'.format(convert_pred[-10:]))

    if diffed is not None:
        f.write("\n\n변화량 비교")
        f.write('\n diffed: {}'.format(diffed[-10:]))
        f.write('\n\n predict: {}'.format(pred.flatten()[-10:]))
    f.close()


from trainer import make_trainer
from model_lstm import LTSF_LSTM, lstm_model
import torch
import torch.nn as nn


def train_all(file_manager, preprocessor, input_window, output_window, hidden_size, dropout_rate, learning_rate, num_layers, eval_mode=False):
    values = preprocessor.processed(input_window, output_window)
    trainer = make_trainer(file_manager, values)

    if file_manager is not None:
        file_manager.set_params(input_window, output_window, hidden_size, learning_rate, dropout_rate)

    model = lstm_model(
        output_window=output_window,
        feature_size=preprocessor.feature_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_layers=num_layers
    ).to(Const.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    valid_loss = None
    if eval_mode is False:
        valid_loss = trainer.train(Parser.param_epochs, model, criterion, optimizer)
    return valid_loss