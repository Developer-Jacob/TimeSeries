from StockData import ExampleDataset
from torch.utils.data import DataLoader
import Parser


def data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y):

    train_data_set = ExampleDataset(train_x, train_y)
    valid_data_set = ExampleDataset(valid_x, valid_y)
    test_data_set = ExampleDataset(test_x, test_y)

    train_loader = DataLoader(train_data_set, Parser.param_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_data_set, len(valid_data_set), shuffle=False)
    test_loader = DataLoader(test_data_set, len(test_data_set), shuffle=False)

    return train_loader, valid_loader, test_loader