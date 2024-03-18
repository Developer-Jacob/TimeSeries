from sklearn.preprocessing import MinMaxScaler
from pykrx import stock
import numpy as np
import gc

class RowData:
    def __init__(self, input_window, output_window, start="20010101", end="20211231"):
        self.scaler = MinMaxScaler()
        inner_scaler = MinMaxScaler()

        stock_data = stock.get_index_ohlcv_by_date(start, end, "1001")

        self.all_data = stock_data['시가'].to_numpy().reshape(-1, 1)

        train_array = []
        for type in ['고가', '저가', '종가', '거래량']:
            train = stock_data[type][:-output_window].to_numpy().reshape(-1, 1)
            train = inner_scaler.fit_transform(train)
            train_array.append(train)

        train_s = self.scaler.fit_transform(stock_data['시가'][:-output_window].to_numpy().reshape(-1, 1))

        train_array.append(train_s)

        self.train_data = np.array(train_array, dtype=np.float).reshape(5, -1).transpose((1, 0))
        self.test_data = self.train_data[-input_window:].reshape(1, -1, 5)

        del inner_scaler
        del train_s
        del train_array
        del stock_data
        gc.collect()
