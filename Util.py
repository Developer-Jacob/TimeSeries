import matplotlib.pyplot as plt


def draw_result(real, result, path, section=100):
    fig = plt.figure(figsize=(20, 5))
    start = len(real) - section
    if start < 0:
        start = 0
    end = len(real) - 1
    plt.plot(range(start, end), result[start:end], 'b.-')
    plt.plot(range(start, end), real[start:end], 'r.-')
    plt.savefig(path)


def draw_variance(diffed, pred, path, section=100):
    fig = plt.figure(figsize=(20, 5))
    start = len(diffed) - section
    if start < 0:
        start = 0
    end = len(diffed) - 1
    plt.plot(range(start, end), pred[start:end], 'b.-')
    plt.plot(range(start, end), diffed[start:end], 'r.-')
    plt.savefig(path)

def print_result(path, real, diffed, pred, convert_pred):
    f = open(path, 'a+')
    f.write("\n\n값 비교")
    f.write('\n real: {}'.format(real[-10:]))
    f.write('\n\n convert pred: {}'.format(convert_pred[-10:]))
    f.write("\n\n변화량 비교")
    f.write('\n diffed: {}'.format(diffed[-10:]))
    f.write('\n\n predict: {}'.format(pred.flatten()[-10:]))
    f.close()


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