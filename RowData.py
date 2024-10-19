from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch

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

def pct_change(data):
    np.diff(data) / data[:-1]

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

def sliding_on_off(data, target, input_window, output_window, stride=1, diff=False):
    data = np.transpose(np.diff(np.transpose(data))) / data[:-1] * 100
    target = np.diff(target) / target[..., :-1] * 100
    target = np.where(target > 0, 1, 0)
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

    return X, Y

def sliding(input_data, input_target, input_window, output_window, stride=1, diff=False):
    if diff:
        data = np.transpose(np.diff(np.transpose(input_data))) / input_data[:-1] * 100
        target = np.diff(input_target) / input_target[..., :-1] * 100
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

def split_data(data, train_ratio=0.8):
    # Count
    total_count = len(data)
    train_count = int(total_count * train_ratio)
    valid_count = int((total_count - train_count) / 2)
    test_count = total_count - train_count - valid_count

    train_target = data[:train_count]
    valid_target = data[train_count:train_count + valid_count]
    test_target = data[-test_count:]

    return train_target, valid_target, test_target

def normalize(data, target):
    main_scaler = MinMaxScaler()

    # data transform
    new_data = data.reshape(data.shape[1], data.shape[0])
    result = []
    for index in range(len(new_data)):
        if index == 0:
            scaler = main_scaler
        else:
            scaler = MinMaxScaler()
        value = scaler.fit_transform(new_data[index].reshape(-1, 1)).reshape(-1)
        result.append(value)
    result = np.array(result).transpose(1, 0).copy()

    # target transform
    new_target = main_scaler.transform(target.reshape(-1, 1)).reshape(-1)

    return result, new_target, scaler

# class RowData:
#     def __init__(self, input_window, output_window, target_type, types, scaler, train_ratio=0.95):
#         df = fdr.DataReader('KS11', '1995')
#         # Open, High, Low, Close, Volume, Change, UpDown, Comp, Amount, MarCap
#         drop_columns = [name for name in df.columns if name not in types]
#
#         target_data = df[target_type]
#         all_data = df.drop(columns=drop_columns)
#
#         total_count = len(all_data)
#         train_count = int(total_count * train_ratio)
#         test_count = total_count - train_count
#
#         if scaler is not None:
#             fit = target_data[:train_count].to_numpy().reshape(-1, 1)
#             scaler.fit(fit)
#             target = scaler.transform(target_data.to_numpy().reshape(-1, 1))
#             target = target.reshape(1, -1)[0]
#         else:
#             target = target_data.to_numpy()
#
#         base = []
#         # Scaling data types
#         for type in types:
#             if scaler is not None:
#                 inner_scaler = MinMaxScaler()
#
#                 fit = all_data[:train_count][type].to_numpy().reshape(-1, 1)
#                 inner_scaler.fit(fit)
#                 value = inner_scaler.transform(all_data[type].to_numpy().reshape(-1, 1))
#                 value = value.reshape(1, -1)[0]
#                 base.append(value)
#             else:
#                 base.append(all_data[type].to_numpy())
#
#         base = np.array(base).transpose(1, 0)
#
#         target_data[:train_count].to_numpy().reshape(-1, 1)
#
#         train_target = target[:train_count]
#         test_target = target[-test_count:]
#         train_data = base[:train_count]
#         test_data = base[-test_count:]
#         self.real_data = df[target_type][-test_count:].to_numpy().reshape(-1, 1)
#
#         comp = target_data.to_numpy()-df['Comp'].to_numpy()
#         comp_target = comp[:train_count]
#         comp_test = comp[-test_count:]
#
#         self.train_x, self.train_y, self.train_c = sliding(
#             train_target,
#             comp_target,
#             train_data,
#             input_window,
#             output_window
#         )
#         self.test_x, self.test_y, self.test_c = sliding(
#             test_target,
#             comp_test,
#             test_data,
#             input_window,
#             output_window
#         )
#
#         # train: 7036, feature
#         # test: 371, feature
#         # real: 371, 1
#         print(
#             'train_data:', train_data.shape,
#             'test_data:', test_data.shape,
#             'real_data:', self.real_data.shape
#         )
