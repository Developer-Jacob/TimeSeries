import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import torch
from AEModel import StackedAutoEncoder

from torch.utils.data import Dataset

class ExampleDataset(Dataset):
    def __init__(self, x, y):
        super(ExampleDataset, self).__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class StockData:
    def __init__(self, real, train, valid, test, train_target, valid_target, test_target):
        self.real = real
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.train_target = train_target
        self.valid_target = valid_target
        self.test_target = test_target


class StockDataGenerator:
    def FFT(self, data, topn=20):
        fft = np.fft.fft(data)
        fft[topn:-topn] = 0
        ifft = np.fft.ifft(fft)
        return abs(ifft)

    def augment(self, df):
        # Train

        # s = MinMaxScaler()
        # df['OpenRatio'] = s.fit_transform(((df['Open'] - df['Close']) / df['Close']).to_numpy().reshape(-1, 1))
        # df['LowRatio'] = s.transform(((df['Low'] - df['Close']) / df['Close']).to_numpy().reshape(-1, 1))
        # df['HighRatio'] = s.transform(((df['High'] - df['Close']) / df['Close']).to_numpy().reshape(-1, 1))
        # df['MA'] = df['NormClose'].rolling(window=20).mean()
        # std = df['NormClose'].rolling(window=20).std()
        # df['Upper'] = df['MA'] + (2 * std)  # 상단밴드
        # df['Lower'] = df['MA'] - (2 * std)  # 하단밴드
        #
        # windows = [5, 10, 20, 60, 120]
        #
        # for window in windows:
        #     # Normalization된 종가
        #     df['NormCloseEMA{}'.format(window)] = (df['NormClose']
        #                                            .ewm(span=window, min_periods=window, adjust=False)
        #                                            .mean()
        #                                            )
        #     # log 변환되기 전의 거래량
        #     df['VolumeEMA{}'.format(window)] = (df['Volume']
        #                                         .ewm(span=window, min_periods=window, adjust=False)
        #                                         .mean()
        #                                         )
        #     # log 변환
        #     df['VolumeLogEMA{}'.format(window)] = df['VolumeEMA{}'.format(window)].apply(
        #         lambda x: (math.log10(x / 10000000)) if x / 10000000 > 1 else 0)
        #
        close = df['Close'].to_numpy()
        open = df['Open'].to_numpy()
        high = df['High'].to_numpy()
        low = df['Low'].to_numpy()
        #
        # open_ratio = df['OpenRatio'].to_numpy()
        # high_ratio = df['HighRatio'].to_numpy()
        # low_ratio = df['LowRatio'].to_numpy()
        # bollinger_upper = df['Upper'].to_numpy()
        # bollinger_lower = df['Lower'].to_numpy()
        # bollinger_ma = df['MA'].to_numpy()
        # norm_close_ema5 = df['NormCloseEMA5'].to_numpy()
        # norm_close_ema10 = df['NormCloseEMA10'].to_numpy()
        # norm_close_ema20 = df['NormCloseEMA20'].to_numpy()
        # norm_close_ema60 = df['NormCloseEMA60'].to_numpy()
        # norm_close_ema120 = df['NormCloseEMA120'].to_numpy()
        # volume_log_ema5 = df['VolumeLogEMA5'].to_numpy()
        # volume_log_ema10 = df['VolumeLogEMA10'].to_numpy()
        # volume_log_ema20 = df['VolumeLogEMA20'].to_numpy()
        # volume_log_ema60 = df['VolumeLogEMA60'].to_numpy()
        # volume_log_ema120 = df['VolumeLogEMA120'].to_numpy()

        result = [
            close
            # norm_open, norm_high, norm_low,
            # open_ratio, high_ratio, low_ratio,
            # bollinger_upper, bollinger_lower, bollinger_ma,
            # norm_close_ema5, norm_close_ema10, norm_close_ema20, norm_close_ema60, norm_close_ema120,
            # volume_log_ema5, volume_log_ema10, volume_log_ema20, volume_log_ema60, volume_log_ema120,
        ]
        result = np.array(result).transpose(1, 0).copy()

        return result, close

    def generateRowData(self, section_size=600):
        result = []
        for i in range(len(self.data_frame)):
            start = i * section_size
            end = (i + 1) * section_size
            if end >= len(self.data_frame):
                break

            value = self.data_frame[start:end]

            result.append(self.augment(value))

        return result

    def allGenerateData(self):
        return self.generateData(0, len(self.data_frame))

    def generateData(self, start_index, section_size, train_ratio=0.8):
        end_index = start_index + section_size
        data = self.data_frame[start_index:end_index]

        # Count
        total_count = len(data)
        train_count = int(total_count * train_ratio)
        valid_count = int((total_count - train_count) / 2)
        test_count = total_count - train_count - valid_count

        target_data = data[self.target_class]

        # Real
        real = target_data[-test_count:].to_numpy().copy()

        # Scaler fitting

        # scaler.fit(target_data[:train_count].to_numpy().reshape(-1, 1))

        augmented_data, target = self.augment(data)

        train_augmented_data = augmented_data[:train_count]
        valid_augmented_data = augmented_data[train_count:train_count + valid_count]
        test_augmented_data = augmented_data[-test_count:]

        train_target = target[:train_count]
        valid_target = target[train_count:train_count + valid_count]
        test_target = target[-test_count:]

        stock_data = StockData(
            real,
            train_augmented_data,
            valid_augmented_data,
            test_augmented_data,
            train_target,
            valid_target,
            test_target,
        )

        return stock_data

    def __init__(self, target_class='Close'):
        self.target_class = target_class
        self.data_frame = fdr.DataReader('S&P500', '1985').copy()
        print('Total Data length:', len(self.data_frame))
        self.total_data_size = len(self.data_frame)