import Parser
from trainer import Trainer
from Const import device
from StockData import StockDataGenerator
from FileManager import FileManager
from DataLoader import data_loader
from Preprocessor import Preprocessor
from Student import Student
from model_lstm import LTSF_LSTM
import torch
import torch.nn as nn
from Util import draw_result, print_result
import numpy as np


def main():
    print("Device: ", device)
    file_manager = FileManager()
    print("--------------------------- STEP 1 DATA GENERATOR --------------------")
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()  # ndarray
    # data_set = generator.dummy()

    mode = "study"

    epochs = Parser.param_epochs
    preprocessor = Preprocessor(data_set, need_diff=True, need_norm=True, verbose=False)
    print("--------------------------- STEP 2 TRAINING MODE: ", mode, "--------------------")

    pred = None
    input_window = None
    if mode == "study":
        student = Student(epochs, preprocessor, _make_trainer, _make_model)
        student.study()
        lstm_model, trainer = student.train_with_best_params()
        pred = trainer.eval(lstm_model)
        file_manager.set_params(
            student.best_input_window(),
            1,
            student.best_hidden_size(),
            student.best_learning_rate(),
            student.best_dropout_rate()
        )
        input_window = student.best_input_window()
    elif mode == "train":
        # input_window = Parser.param_input_window
        input_window = 4
        output_window = Parser.param_output_window
        # hidden_size = Parser.param_hidden_size
        hidden_size = 32
        # learning_rate = Parser.param_learning_rate
        learning_rate = 0.0286
        # dropout = 0.2
        dropout = 0.16
        model, optimizer, criterion = _make_model(output_window, hidden_size, dropout, learning_rate)
        trainer = _make_trainer(preprocessor, input_window, output_window)
        trained_model, valid_loss = trainer.train(Parser.param_epochs, model, criterion, optimizer)
        file_manager.set_params(input_window, output_window, hidden_size, learning_rate, dropout)
        file_manager.save_model(trained_model)
        pred = trainer.eval(trained_model)
    if pred is None:
        print("!! No prediction")
        return
    pred = pred[:, :, 0]

    pred = preprocessor.inverse_normalize_test_target(pred)

    real = data_set.test_target
    output = []
    for index, _ in enumerate(real):
        diff_index = index - input_window - 1
        if len(pred) <= diff_index:
            break
        if diff_index < 0:
            output.append(0)
            continue
        else:
            data = real[index - 1] * (1 + (pred[diff_index]/100))
            output.append(data[0])
    output = np.array(output)
    draw_result(real, np.array(output), file_manager.image_path)
    print_result(file_manager.file_path, real, preprocessor.diffed()[5], pred, np.array(output))
    print("Completed draw, print.")


def _make_trainer(_preprocessor, _input_windows, _output_windows):
    _train_x, _train_y, _valid_x, _valid_y, _test_x, _test_y = _preprocessor.processed(_input_windows, _output_windows)
    _train_loader, _valid_loader, _test_loader = data_loader(_train_x, _train_y, _valid_x, _valid_y, _test_x, _test_y)
    return Trainer(_train_loader, _valid_loader, _test_loader)


def _make_model(_output_window, _hidden_size, _drop_out, _learning_rate):
    _lstm_model = LTSF_LSTM(
        _output_window,
        feature_size=Parser.feature_size,
        hidden_size=_hidden_size,
        dropout=_drop_out
    ).to(device)

    _optimizer = torch.optim.Adam(_lstm_model.parameters(), lr=_learning_rate)
    _criterion = nn.MSELoss()
    return _lstm_model, _optimizer, _criterion


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    main()