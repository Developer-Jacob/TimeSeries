import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset

import Parser
from trainer import Trainer
from RowData import sliding, normalize, split_data, prepare_data, to_tensor
from model_lstm import LTSF_LSTM
from Const import device
from StockData import StockData, StockDataGenerator, ExampleDataset
from AEModel import CnnAutoEncoder, StackedAutoEncoder
from Util import path, make_file, drow_loss, show, showPLT, collate_fn, showTemp, printt


def prepare(data_set, need_encode):
    input_window = Parser.param_input_window
    if need_encode:
        n_epoch = Parser.param_encoder_epochs

        encoder = StackedAutoEncoder(n_epoch=n_epoch)

        # Train Autoencoder
        encoder.forward(data_set.train_data)

        # Encode data
        encoded_train_data = encoder.encoded_data(data_set.train_data)  # Numpy
        encoded_valid_data = encoder.encoded_data(data_set.valid_data)  # Numpy
        encoded_test_data = encoder.encoded_data(data_set.test_data)  # Numpy

        extra_encoded_valid_data = np.concatenate((encoded_train_data[-input_window:], encoded_valid_data))
        extra_encoded_test_data = np.concatenate((encoded_valid_data[-input_window:], encoded_test_data))

        x_train, y_train = prepare_data(encoded_train_data, data_set.train_target, input_window, log_return=True, train=True)
        x_valid, y_valid = prepare_data(extra_encoded_valid_data, data_set.valid_target, input_window, log_return=False, train=True)
        x_test, y_test = prepare_data(extra_encoded_test_data, data_set.test_target, input_window, log_return=False, train=True)

        return to_tensor(x_train), to_tensor(y_train), x_valid, y_valid, x_test, y_test
    else:
        x_train, y_train = prepare_data(data_set.train_data, data_set.train_target, input_window, log_return=True, train=True)
        x_valid, y_valid = prepare_data(data_set.valid_data.astype(np.float32), data_set.valid_target.astype(np.float32), input_window, log_return=False, train=True)
        x_test, y_test = prepare_data(data_set.test_data.astype(np.float32), data_set.test_target.astype(np.float32), input_window, log_return=False, train=True)

        return (to_tensor(np.array(x_train)),
                to_tensor(np.expand_dims(np.array(y_train), axis=-1)),
                to_tensor(np.array(x_valid)),
                to_tensor(np.expand_dims(np.array(y_valid), axis=-1)),
                to_tensor(np.array(x_test)),
                to_tensor(np.expand_dims(np.array(y_test), axis=-1)),
                )


def main():
    need_normalize = False

    # Parser
    epochs = Parser.param_epochs
    input_window = Parser.param_input_window
    output_window = Parser.param_output_window
    hidden_size = Parser.param_hidden_size
    learning_rate = Parser.param_learning_rate

    # ---------------------------------------------------------------------------
    # --------------------------- STEP 0: LOAD DATA -----------------------------
    # ---------------------------------------------------------------------------
    print("STEP 1")
    generator = StockDataGenerator()

    data_set = generator.allGenerateData()  # ndarray
    # data_set = generator.dummy()
    # ---------------------------------------------------------------------------
    # ------------- STEP 2: NORMALIZE DATA --------------------------------------
    # ---------------------------------------------------------------------------
    print("STEP 2")
    # if need_normalize:
    #     data, target, scaler = normalize(data, target)

    # ---------------------------------------------------------------------------
    # ------------- STEP 3: ENCODE FEATURES USING STACKED AUTOENCODER -----------
    # ---------------------------------------------------------------------------
    print("STEP 3")
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare(data_set, need_encode=False)

    # ---------------------------------------------------------------------------
    # ------------- STEP 4: MAKE MODEL ------------------------------------------
    # ---------------------------------------------------------------------------
    print("STEP 4 MAKE MODEL")
    feature_size = train_x.shape[2]
    lstm_model = LTSF_LSTM(input_window, output_window, feature_size=feature_size, hidden_size=hidden_size).to(device)

    # ---------------------------------------------------------------------------
    # ------------- STEP 5: MAKE DATALOADER -------------------------------------
    # ---------------------------------------------------------------------------
    print("STEP 5 MAKE DATALOADER")
    train_data_set = ExampleDataset(train_x, train_y)
    valid_data_set = ExampleDataset(valid_x, valid_y)
    test_data_set = ExampleDataset(test_x, test_y)

    train_loader = DataLoader(train_data_set, 64, shuffle=True)
    valid_loader = DataLoader(valid_data_set, len(valid_data_set), shuffle=False)
    test_loader = DataLoader(test_data_set, len(test_data_set), shuffle=False)

    # ---------------------------------------------------------------------------
    # ------------- STEP 6: TRAINING --------------------------------------------
    # ---------------------------------------------------------------------------
    print("STEP 6 TRAINING")
    # Train
    trainer = Trainer(train_loader, valid_loader, test_loader)
    criterion = nn.MSELoss()
    param = lstm_model.parameters()
    optimizer = torch.optim.Adam(param, lr=learning_rate)
    loss_train, loss_valid, loss_test = trainer.train(epochs, lstm_model, criterion, optimizer)

    # ---------------------------------------------------------------------------
    # ------------- STEP 7: SHOW RESULT --------------------------------------------
    # ---------------------------------------------------------------------------
    print("STEP 7 SHOW RESULT")
    pred = trainer.eval(lstm_model)
    pred = pred[:, :, 0]


    real = data_set.test_target
    output = []
    for index, _ in enumerate(real):
        diff_index = index - input_window
        if len(pred) <= diff_index:
            break
        if diff_index < 0:
            output.append(0)
            continue
        else:
            data = real[index - 1] * (1 + pred[diff_index])
            output.append(data[0])
    showTemp(real, np.array(output), pred)
    printt(real, np.array(output), test_y, pred)


if __name__ == "__main__":
    print("Device: ", device)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context


    make_file()

    main()