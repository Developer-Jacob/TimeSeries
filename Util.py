from datetime import datetime
import os
import matplotlib.pyplot as plt
import Parser
dir_path = ""
import torch

def collate_fn(batchDummy):
    x = [torch.LongTensor(batch[0])for batch in batchDummy]
    # batch단위로 데이터가 넘어올 때 아래 pad_sequence를 통해 알아서 padding을 해준다
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    return {'x' : x}

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
def showTemp(real, result, compare, section=100, show=False):
    fig = plt.figure(figsize=(20, 5))

    compare = np.insert(compare, 0, [0, 0, 0])
    start = len(real) - section
    if start < 0:
        start = 0
    end = len(real) - 1
    plt.plot(range(start, end), result[start:end], 'b.-')
    plt.plot(range(start, end), real[start:end], 'r.-')

    if show:
        plt.show()
    else:
        plt.savefig(path() + '/result.png')
def printt(real, predit, result, compare):
    f = open(file_path(), 'a+')
    f.write('\n\n real: {}'.format(real[-10:]))
    f.write('\n\n predict: {}'.format(predit[-10:]))
    f.write('\n\n real compare: {}'.format(result[-10:]))
    f.write('\n\n predict compare: {}'.format(compare[-10:]))
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
#
# def show_data(data_generator):
#     plt.figure(figsize=(20,10))
#     data = data_generator.test.transpose(1, 0)
#
#     plt.subplot(2, 3, 1)
#     for i in range(0, 4):
#         plt.plot(data[i])
#     plt.subplot(2, 3, 2)
#     for i in range(4, 7):
#         plt.plot(data[i])
#     plt.subplot(2, 3, 3)
#     for i in range(7, 10):
#         plt.plot(data[i])
#     plt.subplot(2, 3, 4)
#     for i in range(10, 15):
#         plt.plot(data[i])
#     plt.subplot(2, 3, 5)
#     for i in range(15, 20):
#         plt.plot(data[i])
#     plt.subplot(2, 3, 6)
#     for i in range(0, len(data)):
#         plt.plot(data[i])
#     plt.show()