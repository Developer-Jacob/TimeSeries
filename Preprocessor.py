from AEModel import CnnAutoEncoder, StackedAutoEncoder
import Parser
import numpy as np
import torch
from Differ import diff_data, diff_target


def to_tensor(array):
    return torch.tensor(array).to(dtype=torch.float32)


def shift_elements(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def prepare_data(x_encoded, y_close, time_steps, log_return=True, train=True):
    ct = 0
    data = []
    for i in range(len(x_encoded) - time_steps):
        ct += 1
        if train:
            x_train = x_encoded[i:i+time_steps]
        else:
            x_train = x_encoded[:i+time_steps]

        data.append(x_train)

    if log_return == False:
        y_close = np.diff(y_close) / y_close[..., :-1]
        y_close = np.float32(y_close)
        # y_close = pct_change(y_close)
    else:
        # y_close = (np.log(y_close) - np.log(y_close.shift(1)))[1:] # the log return, i.e. ln(y_t/y_(t-1))
        shifted = shift_elements(y_close, 1, 0)
        y_close = (np.log(y_close) - np.log(shifted))[1:]  # the log return, i.e. ln(y_t/y_(t-1))
    if train:
        y = y_close[time_steps-1:]
    else:
        y = y_close

    result = zip(data, y)
    return zip(*result)


def prepare(data_set, need_encode):
    input_window = Parser.param_input_window
    output_window = Parser.param_output_window
    if need_encode:
        n_epoch = Parser.param_encoder_epochs

        encoder = StackedAutoEncoder(n_epoch=n_epoch)

        # Train Autoencoder
        encoder.forward(data_set.train_data)

        # Encode data
        encoded_train_data = encoder.encoded_data(data_set.train_data)  # Numpy
        encoded_valid_data = encoder.encoded_data(data_set.valid_data)  # Numpy
        encoded_test_data = encoder.encoded_data(data_set.test_data)  # Numpy

        extra_encoded_valid_data = np.concatenate((encoded_train_data[-input_window:], encoded_valid_data))
        extra_encoded_test_data = np.concatenate((encoded_valid_data[-input_window:], encoded_test_data))

        x_train, y_train = prepare_data(encoded_train_data, data_set.train_target, input_window, log_return=False, train=True)
        x_valid, y_valid = prepare_data(extra_encoded_valid_data, data_set.valid_target, input_window, log_return=False, train=True)
        x_test, y_test = prepare_data(extra_encoded_test_data, data_set.test_target, input_window, log_return=False, train=True)

        return to_tensor(x_train), to_tensor(y_train), x_valid, y_valid, x_test, y_test
    else:
        diffed_train_x = diff_data(data_set.train_data)
        diffed_train_y = diff_target(data_set.train_target)
        diffed_valid_x = diff_data(data_set.valid_data)
        diffed_valid_y = diff_target(data_set.valid_target)
        diffed_test_x = diff_data(data_set.test_data)
        diffed_test_y = diff_target(data_set.test_target)

        x_train, y_train = sliding(data_set.train_data, data_set.train_target, input_window, output_window, diff=True)
        x_valid, y_valid = sliding(data_set.valid_data.astype(np.float32), data_set.valid_target.astype(np.float32), input_window, output_window, diff=True)
        x_test, y_test = sliding(data_set.test_data.astype(np.float32), data_set.test_target.astype(np.float32), input_window, output_window, diff=True)

        tensor_x_train = to_tensor(x_train)
        tensor_y_train = to_tensor(y_train)
        tensor_x_valid = to_tensor(x_valid)
        tensor_y_valid = to_tensor(y_valid)
        tensor_x_test = to_tensor(x_test)
        tensor_y_test = to_tensor(y_test)

        print("Train X:   ", tensor_x_train.shape, tensor_x_train[0])
        print("Train Y:   ", tensor_y_train.shape, tensor_y_train[0])
        print("Valid X:   ", tensor_x_valid.shape, tensor_x_valid[0])
        print("Valid Y:   ", tensor_y_valid.shape, tensor_y_valid[0])
        print("Test X:    ", tensor_x_test.shape, tensor_x_test[0])
        print("Test Y:    ", tensor_y_test.shape, tensor_y_test[0])

        return tensor_x_train, tensor_y_train, tensor_x_valid, tensor_y_valid, tensor_x_test, tensor_y_test


def sliding(input_data, input_target, input_window, output_window, stride=1, diff=False):
    if diff:
        diffed = np.transpose(np.diff(np.transpose(input_data)))
        pre_input = input_data[:-1]
        data = np.where(pre_input != 0, diffed/pre_input, 0) * 100
        # data2 = np.transpose(np.diff(np.transpose(input_data))) / input_data[:-1] * 100
        deffed2 = np.diff(input_target)
        pre_input2 = input_target[..., :-1]
        target = np.where(pre_input2 != 0, deffed2/pre_input2, 0) * 100
    else:
        data = input_data
        target = input_target
    # 데이터의 개수
    L = data.shape[0]
    feature_size = data.shape[1]
    # stride씩 움직이는데 몇번움직임 가능한지
    num_samples = (L - input_window - output_window + 1) // stride

    # input, output
    X = np.zeros([num_samples, input_window, feature_size])
    Y = np.zeros([num_samples, output_window])

    for i in np.arange(num_samples):
        start_x = stride * i
        end_x = start_x + input_window

        start_y = stride * i + input_window
        end_y = start_y + output_window

        X[i] = data[start_x:end_x]
        Y[i] = target[start_y:end_y]

    return X, Y.squeeze()