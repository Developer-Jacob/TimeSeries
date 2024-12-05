from datetime import datetime
import os
import matplotlib.pyplot as plt
import Parser
import torch

dir_path = ""

def path():
    date_str = datetime.today().strftime("%Y%m%d")
    directory = 'EP{}_IW{}_OW{}_HS{}_LR{}'.format(
        Parser.param_epochs,
        Parser.param_input_window,
        Parser.param_output_window,
        Parser.param_hidden_size,
        Parser.param_learning_rate
    )
    return './Model/{}/{}'.format(date_str, directory)

def file_path():
    return path() + '/result.txt'

def make_file():
    os.makedirs(path(), exist_ok=True)
    f = open(file_path(), 'w+')
    f.write('Epochs {}'.format(Parser.param_epochs))
    f.write('\nInputWindow {}'.format(Parser.param_input_window))
    f.write('\nOutputWindow {}'.format(Parser.param_output_window))
    f.write('\nHiddenSize {}'.format(Parser.param_hidden_size))
    f.write('\nLearningRate {}'.format(Parser.param_learning_rate))
    f.close()

def drow_loss(train, valid, test):
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(train)), train, 'b')
    plt.plot(range(0, len(valid)), valid, 'r')
    plt.plot(range(0, len(test)), test, 'k')
    plt.savefig(path() + '/loss.png')
    plt.show()

def showPLT(data, encoded):
    fig = plt.figure(figsize=(20, 5))

    plt.plot(range(0, len(data)), data, 'red')
    plt.plot(range(0, len(encoded)), encoded, 'blue')

    plt.show()

import numpy as np
def showTemp2(real, pred):
    real_one_zero = np.where(np.diff(real[15:]) > 0, 1, 0)

def show_new(list, term=0, section=100):
    fig = plt.figure(figsize=(20, 5))
    for index, data in enumerate(list):
        show_data = np.where(len(data) < section, data, data[-section:])
        start = index * term
        end = start + len(show_data)
        plt.plot(range(start, end), show_data, '.-')


def showTemp(real, result, section=100, file_name="result.png"):
    fig = plt.figure(figsize=(20, 5))

    start = len(real) - section
    if start < 0:
        start = 0
    end = len(real) - 1
    plt.plot(range(start, end), result[start:end], 'b.-')
    plt.plot(range(start, end), real[start:end], 'r.-')
    plt.savefig(path() + "/" + file_name)


def show_on_off(real, pred, section=100):
    fig = plt.figure(figsize=(20, 5))
    start = len(real) - section

    if show:
        plt.show()
    else:
        plt.savefig(path() + '/result.png')
def printt(real, predit, result, compare):
    f = open(file_path(), 'a+')
    f.write('\n\n real: {}'.format(real[-10:-1]))
    f.write('\n\n predict: {}'.format(predit[-9:]))
    f.write('\n\n real compare: {}'.format(result[-10:-1]))
    f.write('\n\n predict compare: {}'.format(compare[-9:]))
    f.close()

def show(real, result, input_window, output_window, show=False):
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
        plt.savefig(path() + '/result.png')