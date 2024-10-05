import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import Parser
import Util
from trainer import Trainer
from RowData import sliding, normalize, split_data, prepare_data, to_tensor
from model_lstm import LTSF_LSTM
from Const import device

import numpy as np
import FinanceDataReader as fdr
from StockData import StockData, StockDataGenerator, ExampleDataset
from torch.utils.data import TensorDataset, DataLoader, Dataset
from AEModel import CnnAutoEncoder, StackedAutoEncoder
from Normalizer import Normalizer
from Util import path, make_file, drow_loss, show, showPLT


def main():
    make_file()

# * 인풋이 분봉 -> 아웃풋이 일봉
# * 아웃풋 3일간의 고점 저점 예측

    need_normalize = False
    epochs = Parser.param_epochs
    encoder_epochs = Parser.param_encoder_epochs
    input_window = Parser.param_input_window
    output_window = Parser.param_output_window
    hidden_size = Parser.param_hidden_size
    learning_rate = Parser.param_learning_rate

    # ---------------------------------------------------------------------------
    # --------------------------- STEP 0: LOAD DATA -----------------------------
    # ---------------------------------------------------------------------------
    generator = StockDataGenerator()
    data_list = generator.generateRowData(section_size=600)
    first_data, _ = data_list[0]
    feature_size = first_data[1].size
    encoder = StackedAutoEncoder(n_epoch=encoder_epochs)

    # ---------------------------------------------------------------------------
    # --------------------------- STEP 0: MAKE MODEL ----------------------------
    # ---------------------------------------------------------------------------
    lstm_model = LTSF_LSTM(input_window, output_window, feature_size=10, hidden_size=hidden_size).to(device)

    for (data, target) in data_list:

        # ---------------------------------------------------------------------------
        # ----------------------- STEP 2.0: NORMALIZE DATA --------------------------
        # ---------------------------------------------------------------------------
        if need_normalize:
            data, target, scaler = normalize(data, target)

        # Split data
        # x
        train_data, valid_data, test_data = split_data(data)
        # y
        train_target, valid_target, test_target = split_data(target)

        # ---------------------------------------------------------------------------
        # ------------- STEP 3: ENCODE FEATURES USING STACKED AUTOENCODER -----------
        # ---------------------------------------------------------------------------
        # Train Autoencoder
        encoder.forward(train_data)

        # Encode data
        encoded_train_data = encoder.encoded_data(train_data)   # Numpy
        encoded_valid_data = encoder.encoded_data(valid_data)   # Numpy
        encoded_test_data = encoder.encoded_data(test_data)     # Numpy

        extra_encoded_valid_data = np.concatenate((encoded_train_data[-input_window:], encoded_valid_data))
        extra_encoded_test_data = np.concatenate((encoded_valid_data[-input_window:], encoded_test_data))

        x_train, y_train = prepare_data(encoded_train_data, train_target, input_window, log_return=True, train=True)
        x_valid, y_valid = prepare_data(extra_encoded_valid_data, valid_target, input_window, log_return=False, train=False)
        x_test, y_test = prepare_data(extra_encoded_test_data, test_target, input_window, log_return=False, train=False)

        # To Tensor
        train_x = to_tensor(x_train)
        train_y = to_tensor(y_train)
        valid_x = x_valid
        valid_y = y_valid
        test_x = x_test
        test_y = y_test

        # Make data loader
        train_data_set = ExampleDataset(train_x, train_y)
        valid_data_set = ExampleDataset(valid_x, valid_y)
        test_data_set = ExampleDataset(test_x, test_y)

        train_loader = DataLoader(train_data_set, 64, shuffle=False)
        valid_loader = DataLoader(valid_data_set, 1, shuffle=False)
        test_loader = DataLoader(test_data_set, 1, shuffle=False)

        # Train
        trainer = Trainer(train_loader, valid_loader, test_loader)
        criterion = nn.MSELoss()
        param = lstm_model.parameters()
        optimizer = torch.optim.Adam(param, lr=learning_rate)
        loss_train, loss_valid, loss_test = trainer.train(epochs, lstm_model, criterion, optimizer)

        result = trainer.eval(lstm_model)
        result = result[:, :, 0]
        # result = scaler.inverse_transform(result)

        real = data.real

        show(real, result, input_window, output_window)

if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    main()