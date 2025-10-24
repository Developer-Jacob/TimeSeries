import pandas as pd
import numpy as np
from StockData import StockData
import ReadExcel


class StockDataGenerator:
    target_class = "Close"
    data_class = []
    feature_size = 0

    def augment(self, df):
        df = df.copy()
        df['MA'] = df['Close'].rolling(window=20).mean()

        std = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA'] + (2 * std)  # 상단밴드
        df['Lower'] = df['MA'] - (2 * std)  # 하단밴드

        def value(key):
            return df[key].to_numpy()

        self.data_class = [
            "Close", "High", "Low", "Volume",
            "Upper", "Lower", "MA"
        ]
        result = list(map(value, self.data_class))
        close = df['Close'].to_numpy()

        self.feature_size = len(result)
        result = np.array(result).transpose(1, 0).copy()
        return result[20:], close[20:]

    def all(self):
        data_set = self.generate(0, len(self.data_frame))
        print_data_set(self.data_class, self.target_class, data_set)
        return data_set

    def half(self):
        data_set = self.generate(0, len(self.data_frame)//2)
        print_data_set(self.data_class, self.target_class, data_set)
        return data_set

    def mini(self):
        data_set = self.generate(0, 1000)
        print_data_set(self.data_class, self.target_class, data_set)
        return data_set

    def generate(self, start_index, section_size, train_ratio=0.8):
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


    def __init__(self):
        self.data_frame = xbt_usd_min()
        print('Total data length:', len(self.data_frame))


def xbt_usd_min(size=0.8):
    read_data_frame = ReadExcel.read_xbtusd_five_to_dataframe().copy()
    return read_data_frame.iloc[int(len(read_data_frame) * size):]


def print_data_set(data_class, target_class, data_set):
    print("data class:         ", data_class)
    print("Target class:       ", target_class)
    print("Train data shape:   ", data_set.train_data.shape)
    print("Train target shape: ", data_set.train_target.shape)
    print("Valid data shape:   ", data_set.valid_data.shape)
    print("Valid target shape: ", data_set.valid_target.shape)
    print("Test data shape:    ", data_set.test_data.shape)
    print("Test target shape:  ", data_set.test_target.shape)


if __name__ == '__main__':
    generator = StockDataGenerator()
    generator.all()