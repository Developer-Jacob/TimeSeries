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
    def __init__(self, data_set, feature_size, need_diff=True, need_norm=True, verbose=False):
        self.data_set = data_set
        self.verbose = verbose
        self.data_normalizer = Normalizer()
        self.target_normalizer = Normalizer()
        self.feature_size = feature_size
        if need_diff:
            values = self.diffed()
        else:
            values = self.raw()

        # list_values = []
        # for i in range(0, len(self.diffed())):
        #     _raw = np.transpose(self.raw()[i][1:], (1, 0))
        #     _diffed = np.transpose(self.diffed()[i], (1, 0))
        #     _value = np.transpose(np.concatenate((_raw, _diffed)), (1, 0))
        #     list_values.append(_value)
        # values = tuple(list_values)
        if need_norm:
            self.processed_value = self.normalized(values)
        else:
            self.processed_value = values
        print("123")

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

    def processed(self, input_window, output_window):
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.processed_value

        x_train, y_train = sliding(train_x, train_y, input_window, output_window)
        x_valid, y_valid = sliding(valid_x.astype(np.float32), valid_y.astype(np.float32), input_window, output_window)
        x_test, y_test = sliding(test_x.astype(np.float32), test_y.astype(np.float32), input_window, output_window)

        tensor_x_train = to_tensor(x_train)
        tensor_y_train = to_tensor(y_train)
        tensor_x_valid = to_tensor(x_valid)
        tensor_y_valid = to_tensor(y_valid)
        tensor_x_test = to_tensor(x_test)
        tensor_y_test = to_tensor(y_test)

        if self.verbose:
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


if __name__ == "__main__":
    from StockData import StockDataGenerator
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()
    preprocessor = Preprocessor(data_set, generator.feature_size, True, False)
    data = preprocessor.processed_value[0]
    data = data.reshape(4, -1)[0]
    data = data.reshape(-1, 1).squeeze()
    import Util

    Util.draw_test(data, None)