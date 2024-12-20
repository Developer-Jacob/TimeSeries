import matplotlib.pyplot as plt
import numpy as np
from CustomLoss import CustomLoss
import Const
import Parser
from Transformer import TimeSeriesTransformer
from Viewer import CustomLinearLoss
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
    f.write("\n\nValues")
    f.write('\nreal: {}'.format(real[-10:]))
    f.write('\n\nconvert pred: {}'.format(convert_pred[-10:]))

    if diffed is not None:
        f.write("\n\nVariance")
        f.write('\ndiffed: {}'.format(diffed[-10:]))
        f.write('\n\npredict: {}'.format(pred.flatten()[-10:]))
    f.close()

import torch
import torch.nn as nn


def train_all(trainer, early_stopping, input_window, output_window, feature_size, hidden_size, dropout_rate, learning_rate, num_layers):
    model = TimeSeriesTransformer(
        input_dim=feature_size,
        d_model=hidden_size,
        n_heads=4,
        num_layers=num_layers,
        seq_len=input_window,
        output_dim=output_window,
        dropout_rate=dropout_rate
    ).to(Const.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.MSELoss()
    criterion = CustomLinearLoss()
    # criterion = nn.SmoothL1Loss()

    valid_loss = trainer.train(early_stopping, Parser.param_epochs, model, criterion, optimizer)
    return valid_loss