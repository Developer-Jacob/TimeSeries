import FinanceDataReader as fdr
import numpy as np
from torch.utils.data import Dataset
import ReadExcel
import yfinance as yf
import pandas as pd

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
    def __init__(self, train, valid, test, train_target, valid_target, test_target):
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.train_target = train_target
        self.valid_target = valid_target
        self.test_target = test_target


class StockDataGenerator:
    def augment(self, df):
        # Train

        df = df.copy()
        df['MA'] = df['Close'].rolling(window=20).mean()


        std = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA'] + (2 * std)  # 상단밴드
        df['Lower'] = df['MA'] - (2 * std)  # 하단밴드

        close = df['Close'].to_numpy()
        open = df['Open'].to_numpy()
        high = df['High'].to_numpy()
        low = df['Low'].to_numpy()
        volume = df['Volume'].to_numpy()

        def value(key):
            return df[key].to_numpy()
            # if key in ["Upper", "Lower", "MA"]:
            #     return df[key].to_numpy()
            # else:
            #     return df[key].rolling(window=5).mean().to_numpy()

        self.data_class = [
            "Close", "High", "Low",
            "Volume",
            "Upper", "Lower", "MA"
        ]
        result = list(map(value, self.data_class))
        # result = [
        #     close, open, high, low,
        #     bollinger_upper, bollinger_lower, bollinger_ma,
        #     # nasdaq
        #     # norm_open, norm_high, norm_low,
        #     # open_ratio, high_ratio, low_ratio,
        #     # bollinger_upper, bollinger_lower, bollinger_ma,
        #     # norm_close_ema5, norm_close_ema10, norm_close_ema20, norm_close_ema60, norm_close_ema120,
        #     # volume_log_ema5, volume_log_ema10, volume_log_ema20, volume_log_ema60, volume_log_ema120,
        # ]

        self.feature_size = len(result)
        result = np.array(result).transpose(1, 0).copy()
        return result[20:], close[20:]

    def allGenerateData(self):
        data_set = self.generateData(0, len(self.data_frame))
        self.print_data_set(data_set)
        return data_set

    def half(self):
        data_set = self.generateData(0, len(self.data_frame)//2)
        self.print_data_set(data_set)
        return data_set

    def mini_data(self):
        data_set = self.generateData(0, 1000)
        self.print_data_set(data_set)
        return data_set

    def print_data_set(self, data_set):
        print("Data class:         ", self.data_class)
        print("Target class:       ", self.target_class)
        print("Train data shape:   ", data_set.train_data.shape)
        print("Train target shape: ", data_set.train_target.shape)
        print("Valid data shape:   ", data_set.valid_data.shape)
        print("Valid target shape: ", data_set.valid_target.shape)
        print("Test data shape:    ", data_set.test_data.shape)
        print("Test target shape:  ", data_set.test_target.shape)

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
            train_augmented_data,
            valid_augmented_data,
            test_augmented_data,
            train_target,
            valid_target,
            test_target,
        )

        return stock_data

    def __init__(self, target_class='Close', merge_count=7):
        self.target_class = target_class
        self.data_frame = xbt_usd_min()
        # self.data_frame = xbt_usd_min(0.1)
        # self.data_frame = merge_data(self.data_frame, merge_count)
        print('Total Data length:', len(self.data_frame))


def xbt_usd_min(size=0.8):
    read_data_frame = ReadExcel.read_xbtusd_five_to_dataframe().copy()
    return read_data_frame.iloc[int(len(read_data_frame) * size):]


def sp500_day():
    data = yf.download('^GSPC', start='1970-01-01', end='2023-12-31').copy()
    data.columns = data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    return data


def merge_data(data_frame, count):
    num_rows = len(data_frame)
    m_close = [None] * num_rows
    m_high = [None] * num_rows
    m_low = [None] * num_rows
    m_open = [None] * num_rows

    # 7개씩 묶어서 처리 (앞에서부터 채우고 마지막 6개는 NaN)
    for i in range(num_rows - count - 1):  # 7개씩 묶기 위해 len(df) - 6 범위 지정
        window = data_frame.iloc[i:i + count]  # 7개 행 슬라이싱
        m_close[i] = window['Close'].iloc[-1]  # 마지막 Close 값
        m_high[i] = window['High'].max()  # High 값 중 최대
        m_low[i] = window['Low'].min()  # Low 값 중 최소
        m_open[i] = window['Open'].iloc[0]  # 첫 Open 값

    data_frame['M_Close'] = m_close
    data_frame['M_High'] = m_high
    data_frame['M_Low'] = m_low
    data_frame['M_Open'] = m_open

    return data_frame


def test_draw_data():
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()
    close = generator.data_frame["Close"].to_numpy()
    total_count = len(close)
    train_count = int(total_count * 0.8)
    valid_count = int((total_count - train_count) / 2)
    test_count = total_count - train_count - valid_count
    import Util
    Util.draw_data_target(
        close[:train_count],
        close[train_count:train_count + valid_count],
        close[-test_count:],
    )


def range_std():
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()
    data = generator.data_frame.to_numpy().copy()
    print("Input data range:", data.min().item(), "-", data.max().item())
    print("Input data standard deviation:", data.std().item())

if __name__ == '__main__':
    print()
