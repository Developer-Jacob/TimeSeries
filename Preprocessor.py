from AEModel import CnnAutoEncoder, StackedAutoEncoder
import numpy as np
import torch
from Differ import diff_data, diff_target
from Normalizer import Normalizer


def to_tensor(array):
    return torch.tensor(array).to(dtype=torch.float32)


def sliding(data, target, input_window, output_window, stride=1):
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

class Preprocessor:
    def __init__(self, data_set, input_window, output_window):
        self.data_set = data_set
        self.input_window = input_window
        self.output_window = output_window
        self.data_normalizer = Normalizer()
        self.target_normalizer = Normalizer()

    def raw(self):
        return (
            self.data_set.train_data,
            self.data_set.train_target,
            self.data_set.valid_data,
            self.data_set.valid_target,
            self.data_set.test_data,
            self.data_set.test_target
        )

    def diffed(self):
        return (
            diff_data(self.data_set.train_data),
            diff_target(self.data_set.train_target),
            diff_data(self.data_set.valid_data),
            diff_target(self.data_set.valid_target),
            diff_data(self.data_set.test_data),
            diff_target(self.data_set.test_target)
        )

    def normalized(self, values):
        train_x, train_y, valid_x, valid_y, test_x, test_y = values
        self.data_normalizer.fit(train_x)
        self.target_normalizer.fit(train_y.reshape(-1, 1))
        return (
            self.data_normalizer.transform(train_x),
            self.target_normalizer.transform(train_y.reshape(-1, 1)).flatten(),
            self.data_normalizer.transform(valid_x),
            self.target_normalizer.transform(valid_y.reshape(-1, 1)).flatten(),
            self.data_normalizer.transform(test_x),
            self.target_normalizer.transform(test_y.reshape(-1, 1)).flatten()
        )

    def inverse_normalized(self, values):
        train_x, train_y, valid_x, valid_y, test_x, test_y = values
        return (
            self.data_normalizer.inverse_transform(train_x),
            self.target_normalizer.inverse_transform(train_y.reshape(-1, 1)).flatten(),
            self.data_normalizer.inverse_transform(valid_x),
            self.target_normalizer.inverse_transform(valid_y.reshape(-1, 1)).flatten(),
            self.data_normalizer.inverse_transform(test_x),
            self.target_normalizer.inverse_transform(test_y.reshape(-1, 1)).flatten()
        )

    def inverse_normalize_test_target(self, test_target):
        return self.target_normalizer.inverse_transform(test_target.reshape(-1, 1))

    def processed(self, need_diff=True, need_normalize=True):
        if need_diff:
            values = self.diffed()
        else:
            values = self.raw()

        if need_normalize:
            train_x, train_y, valid_x, valid_y, test_x, test_y = self.normalized(values)
        else:
            train_x, train_y, valid_x, valid_y, test_x, test_y = values

        x_train, y_train = sliding(train_x, train_y, self.input_window, self.output_window)
        x_valid, y_valid = sliding(valid_x.astype(np.float32), valid_y.astype(np.float32), self.input_window,
                                   self.output_window)
        x_test, y_test = sliding(test_x.astype(np.float32), test_y.astype(np.float32), self.input_window,
                                 self.output_window)

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


# def prepare_data(x_encoded, y_close, time_steps, log_return=True, train=True):
#     ct = 0
#     data = []
#     for i in range(len(x_encoded) - time_steps):
#         ct += 1
#         if train:
#             x_train = x_encoded[i:i+time_steps]
#         else:
#             x_train = x_encoded[:i+time_steps]
#
#         data.append(x_train)
#
#     if log_return == False:
#         y_close = np.diff(y_close) / y_close[..., :-1]
#         y_close = np.float32(y_close)
#         # y_close = pct_change(y_close)
#     else:
#         # y_close = (np.log(y_close) - np.log(y_close.shift(1)))[1:] # the log return, i.e. ln(y_t/y_(t-1))
#         shifted = shift_elements(y_close, 1, 0)
#         y_close = (np.log(y_close) - np.log(shifted))[1:]  # the log return, i.e. ln(y_t/y_(t-1))
#     if train:
#         y = y_close[time_steps-1:]
#     else:
#         y = y_close
#
#     result = zip(data, y)
#     return zip(*result)
#
#
# def prepare(data_set, need_encode):
#     input_window = Parser.param_input_window
#     output_window = Parser.param_output_window
#     if need_encode:
#         n_epoch = Parser.param_encoder_epochs
#
#         encoder = StackedAutoEncoder(n_epoch=n_epoch)
#
#         # Train Autoencoder
#         encoder.forward(data_set.train_data)
#
#         # Encode data
#         encoded_train_data = encoder.encoded_data(data_set.train_data)  # Numpy
#         encoded_valid_data = encoder.encoded_data(data_set.valid_data)  # Numpy
#         encoded_test_data = encoder.encoded_data(data_set.test_data)  # Numpy
#
#         extra_encoded_valid_data = np.concatenate((encoded_train_data[-input_window:], encoded_valid_data))
#         extra_encoded_test_data = np.concatenate((encoded_valid_data[-input_window:], encoded_test_data))
#
#         x_train, y_train = prepare_data(encoded_train_data, data_set.train_target, input_window, log_return=False, train=True)
#         x_valid, y_valid = prepare_data(extra_encoded_valid_data, data_set.valid_target, input_window, log_return=False, train=True)
#         x_test, y_test = prepare_data(extra_encoded_test_data, data_set.test_target, input_window, log_return=False, train=True)
#
#         return to_tensor(x_train), to_tensor(y_train), x_valid, y_valid, x_test, y_test
#     else:
#         diff = True
#
#         input_train_x = data_set.train_data
#         input_train_y = data_set.train_target
#         input_valid_x = data_set.valid_data
#         input_valid_y = data_set.valid_target
#         input_test_x = data_set.test_data
#         input_test_y = data_set.test_target
#
#         if diff:
#             input_train_x = diff_data(data_set.train_data)
#             input_train_y = diff_target(data_set.train_target)
#             input_valid_x = diff_data(data_set.valid_data)
#             input_valid_y = diff_target(data_set.valid_target)
#             input_test_x = diff_data(data_set.test_data)
#             input_test_y = diff_target(data_set.test_target)
#
#         x_train, y_train = sliding(input_train_x, input_train_y, input_window, output_window)
#         x_valid, y_valid = sliding(input_valid_x.astype(np.float32), input_valid_y.astype(np.float32), input_window, output_window)
#         x_test, y_test = sliding(input_test_x.astype(np.float32), input_test_y.astype(np.float32), input_window, output_window)
#
#         tensor_x_train = to_tensor(x_train)
#         tensor_y_train = to_tensor(y_train)
#         tensor_x_valid = to_tensor(x_valid)
#         tensor_y_valid = to_tensor(y_valid)
#         tensor_x_test = to_tensor(x_test)
#         tensor_y_test = to_tensor(y_test)
#
#         print("Train X:   ", tensor_x_train.shape, tensor_x_train[0])
#         print("Train Y:   ", tensor_y_train.shape, tensor_y_train[0])
#         print("Valid X:   ", tensor_x_valid.shape, tensor_x_valid[0])
#         print("Valid Y:   ", tensor_y_valid.shape, tensor_y_valid[0])
#         print("Test X:    ", tensor_x_test.shape, tensor_x_test[0])
#         print("Test Y:    ", tensor_y_test.shape, tensor_y_test[0])
#
#         return tensor_x_train, tensor_y_train, tensor_x_valid, tensor_y_valid, tensor_x_test, tensor_y_test
#
#
