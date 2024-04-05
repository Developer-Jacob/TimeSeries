import FinanceDataReader as fdr
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

def sliding(data, input_window, output_window, stride=1):
    # 데이터의 개수
    L = data.shape[0]
    feature_size = data.shape[1]
    # stride씩 움직이는데 몇번움직임 가능한지
    num_samples = (L - input_window - output_window) // stride + 1

    data_1 = data.transpose((1, 0))[0]  # (5, 3107)
    # input, output
    X = np.zeros([input_window, num_samples, feature_size])
    Y = np.zeros([output_window, num_samples])

    for i in np.arange(num_samples):
        start_x = stride * i
        end_x = start_x + input_window
        X[:, i] = data[start_x:end_x]

        start_y = stride * i + input_window
        end_y = start_y + output_window
        Y[:, i] = data_1[start_y:end_y]

    X = X.reshape(X.shape[0], X.shape[1], feature_size).transpose((1, 0, 2))  # (3012, 72, 5)
    Y = Y.reshape(Y.shape[0], Y.shape[1]).transpose((1, 0))  # (3012, 24)
    return (
        torch.from_numpy(np.array(X)).to(dtype=torch.float32),
        torch.from_numpy(np.array(Y)).to(dtype=torch.float32)
    )

class RowData:
    def __init__(self, input_window, output_window, types, scaler=StandardScaler(), train_ratio=0.95):
        df = fdr.DataReader('KS11', '1995')
        # Open, High, Low, Close, Volume, Change, UpDown, Comp, Amount, MarCap
        drop_columns = [name for name in df.columns if name not in types]
        all_data = df.drop(columns=drop_columns)

        total_count = len(all_data)
        train_count = int(total_count * train_ratio)
        test_count = total_count - train_count

        target_type = 'Close'

        train_value = all_data[:train_count][target_type].to_numpy().reshape(-1, 1)
        scaler.fit(train_value)
        value = all_data[target_type].to_numpy().reshape(-1, 1)
        all_data[target_type] = scaler.transform(value)

        for type in types:
            if target_type == type:
                continue
            inner_scaler = StandardScaler()

            train_value = all_data[:train_count][type].to_numpy().reshape(-1, 1)
            inner_scaler.fit(train_value)
            value = all_data[type].to_numpy().reshape(-1, 1)
            all_data[type] = inner_scaler.transform(value)

        train_data = all_data[:train_count].to_numpy()
        test_data = all_data[-test_count:].to_numpy()
        self.real_data = df[target_type][-test_count:].to_numpy().reshape(-1, 1)

        self.train_x, self.train_y = sliding(train_data, input_window, output_window)
        self.test_x, self.test_y = sliding(test_data, input_window, output_window)
        # train: 7036, feature
        # test: 371, feature
        # real: 371, 1
        print(
            'train_data:', train_data.shape,
            'test_data:', test_data.shape,
            'real_data:', self.real_data.shape
        )
