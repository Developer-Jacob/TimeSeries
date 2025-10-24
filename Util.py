import matplotlib.pyplot as plt
import numpy as np
from CustomLoss import CustomLoss
import Const
import Parser
from Transformer import TimeSeriesTransformer
import Transformer
import Viewer
import FileManager as fm

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

    # plt.plot(range(0, len(train)), train, 'b.-')
    # plt.plot(range(len(train), len(train)+len(valid)), valid, 'r.-')
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


def print_result(path, title, data):
    f = open(path, 'a+')
    f.write("\n\n{}".format(title))
    f.write('\nreal: {}'.format(data[-10:]))

    f.close()


def show_train_log(title, train_losses, valid_losses, test_losses):
    # 학습 후 손실 그래프 출력
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

import torch
import torch.nn as nn


def train_all(trainer, input_window, output_window, feature_size, hidden_size, dropout_rate, learning_rate, num_layers, num_heads):
    model = Transformer.default_model(
        input_window,
        output_window,
        feature_size,
        hidden_size,
        dropout_rate,
        num_layers,
        num_heads
    ).to(Const.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = Viewer.QuantileHuberLoss()
    # criterion = CustomLoss()
    # criterion = Viewer.ImprovedCustomLoss(penalty_weight=0.1, sensitivity=5.0)
    # criterion = Viewer.StockLoss()
    # criterion = nn.SmoothL1Loss()

    loss = trainer.train(Parser.param_epochs, model, criterion, optimizer)
    return loss


def variance_to_origin(real, variance, input_window):
    output = []
    for index, _ in enumerate(real):
        diff_index = index - input_window - 1
        if len(variance) <= diff_index:
            break
        if diff_index < 0:
            output.append(0)
            continue
        else:
            data = real[index - 1] * (1 + (variance[diff_index]/100))
            output.append(data.squeeze())
    return np.array(output)

def draw_upper_lower(path, real, upper_bound, lower_bound, input_window, output_window):
    output_upper = []
    output_lower = []

    for index, _ in enumerate(real):
        diff_index = index - input_window - output_window
        if len(upper_bound) <= diff_index:
            break
        if diff_index < 0:
            output_upper.append(real[index])
            output_lower.append(real[index])
            continue
        else:
            data = real[index - 1] * (1 + (upper_bound[diff_index]/100))
            output_upper.append(data.squeeze())

            data = real[index - 1] * (1 + (lower_bound[diff_index] / 100))
            output_lower.append(data.squeeze())
    output_upper = np.array(output_upper)
    output_lower = np.array(output_lower)
    fig = plt.figure(figsize=(20, 5))
    start = len(real) - 50
    if start < 0:
        start = 0
    end = len(real)
    plt.plot(range(start, end), output_upper[start:end], 'b.-')
    plt.plot(range(start, end), real[start:end], 'r.-')
    plt.plot(range(start, end), output_lower[start:end], 'b.-')
    plt.savefig(path)

def draw_quantiles(real, predictions):
    # 분위수별 예측값
    q10 = predictions[0].squeeze()  # 10% 분위수
    q50 = predictions[1].squeeze()  # 50% 분위수 (중앙값)
    q90 = predictions[2].squeeze()  # 90% 분위수

    real = real[-100:]
    q10 = q10[-100:]
    q50 = q50[-100:]
    q90 = q90[-100:]

    # X 축 (날짜 또는 시간순서)
    timesteps = np.arange(len(real))

    # 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, real, label="Actual", color="black", linewidth=2)
    plt.plot(timesteps, q50, label="Median Prediction (50%)", linestyle="dashed", color="blue")
    plt.fill_between(timesteps, q10, q90, alpha=0.2, color="blue", label="10%-90% Confidence Interval")
    # plt.plot(timesteps, q50, marker='o', linestyle="dashed", color="blue", label="Median Prediction (50%)")
    # plt.fill_between(timesteps, q10, q90, alpha=0.2, color="blue", label="10%-90% Confidence Interval")
    # plt.scatter(timesteps, y_actual[-1], color="red", label="Actual Value", zorder=3)
    plt.ylim(-3, 3)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Transformer Quantile Regression Predictions")
    plt.show()