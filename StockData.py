import FinanceDataReader as fdr
import numpy as np
from torch.utils.data import Dataset
import ReadExcel
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


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