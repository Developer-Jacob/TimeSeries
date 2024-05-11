import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import pybithumb

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
    def augment(self, df, scaler):
        # Train
        df['NormClose'] = scaler.transform(df['Close'].to_numpy().reshape(-1, 1))
        df['OpenRatio'] = (df['Open'] - df['Close']) / df['Close']
        df['LowRatio'] = (df['Low'] - df['Close']) / df['Close']
        df['HighRatio'] = (df['High'] - df['Close']) / df['Close']
        df['MA'] = df['NormClose'].rolling(window=20).mean()
        std = df['NormClose'].rolling(window=20).std()
        df['Upper'] = df['MA'] + (2 * std)  # 상단밴드
        df['Lower'] = df['MA'] - (2 * std)  # 하단밴드

        windows = [5, 10, 20, 60, 120]

        for window in windows:
            # Normalization된 종가
            df['NormCloseEMA{}'.format(window)] = df['NormClose'].ewm(span=window, min_periods=window,
                                                                      adjust=False).mean()
            # log 변환되기 전의 거래량
            df['VolumeEMA{}'.format(window)] = df['Volume'].ewm(span=window, min_periods=window, adjust=False).mean()
            # log 변환
            df['VolumeLogEMA{}'.format(window)] = df['VolumeEMA{}'.format(window)].apply(
                lambda x: (math.log10(x / 1000000)) if x / 1000000 > 1 else 0)

        norm_close = df['NormClose'].to_numpy()
        open_ratio = df['OpenRatio'].to_numpy()
        high_ratio = df['HighRatio'].to_numpy()
        low_ratio = df['LowRatio'].to_numpy()
        bollinger_upper = df['Upper'].to_numpy()
        bollinger_lower = df['Lower'].to_numpy()
        bollinger_ma = df['MA'].to_numpy()
        norm_close_ema5 = df['NormCloseEMA5'].to_numpy()
        norm_close_ema10 = df['NormCloseEMA10'].to_numpy()
        norm_close_ema20 = df['NormCloseEMA20'].to_numpy()
        norm_close_ema60 = df['NormCloseEMA60'].to_numpy()
        norm_close_ema120 = df['NormCloseEMA120'].to_numpy()
        volume_log_ema5 = df['VolumeLogEMA5'].to_numpy()
        volume_log_ema10 = df['VolumeLogEMA10'].to_numpy()
        volume_log_ema20 = df['VolumeLogEMA20'].to_numpy()
        volume_log_ema60 = df['VolumeLogEMA60'].to_numpy()
        volume_log_ema120 = df['VolumeLogEMA120'].to_numpy()
        result = [
            norm_close,
            open_ratio, high_ratio, low_ratio,
            bollinger_upper, bollinger_lower, bollinger_ma,
            norm_close_ema5, norm_close_ema10, norm_close_ema20, norm_close_ema60, norm_close_ema120,
            volume_log_ema5, volume_log_ema10, volume_log_ema20, volume_log_ema60, volume_log_ema120,
        ]
        result = np.array(result).transpose(1, 0).copy()

        return result, df['NormClose'].to_numpy()

    def FFT(self, data, topn=2):
        fft = np.fft.fft(data)
        fft[topn:-topn] = 0
        ifft = np.fft.ifft(fft)
        return ifft

    def __init__(self, train_ratio=0.8):
        data_frame = fdr.DataReader('KS11', '1995')
        print('Data length:', len(data_frame))
        self.scaler = MinMaxScaler()

        total_count = len(data_frame)
        train_count = int(total_count * train_ratio)
        valid_count = int((total_count - train_count) / 2)
        test_count = total_count - train_count - valid_count

        # Real
        self.real_close = data_frame['Close'][-test_count:].to_numpy().copy()
        # Scaler fitting

        self.scaler.fit(data_frame['Close'][:train_count].to_numpy().reshape(-1, 1))

        augmented_data, target = self.augment(data_frame, self.scaler)
        augmented_data, target = augmented_data[120:], target[120:]

        train_count = train_count - 120

        train_augmented_data = augmented_data[:train_count]
        valid_augmented_data = augmented_data[train_count:train_count+valid_count]
        test_augmented_data = augmented_data[-test_count:]

        train_target = target[:train_count]
        valid_target = target[train_count:train_count+valid_count]
        test_target = target[-test_count:]

        self.stock_data = StockData(
            self.real_close,
            train_augmented_data,
            valid_augmented_data,
            test_augmented_data,
            train_target,
            valid_target,
            test_target,
        )