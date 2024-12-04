import Parser
from trainer import Trainer
from Const import device
from StockData import StockDataGenerator
from Util import make_file
from DataLoader import data_loader
from Preprocessor import Preprocessor
from Student import Student
from model_lstm import LTSF_LSTM
import torch
import torch.nn as nn
from Util import showTemp, printt
import numpy as np
def main():
    mode = "study"
    need_normalize = True
    print("Device: ", device)
    print("--------------------------- STEP 0 CONSTANT --------------------------")
    Parser.print_params()

    print("--------------------------- STEP 1 DATA GENERATOR --------------------")
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()  # ndarray
    # data_set = generator.dummy()

    print("--------------------------- STEP 2 PREPARE DATA ----------------------")
    preprocessor = Preprocessor(data_set, Parser.param_input_window, Parser.param_output_window)
    values = preprocessor.processed(need_diff=True, need_normalize=need_normalize)
    train_x, train_y, valid_x, valid_y, test_x, test_y = values

    print("--------------------------- STEP 3 MAKE MODEL ------------------------")
    # feature_size = train_x.shape[2]

    print("--------------------------- STEP 4 MAKE DATALOADER -------------------")
    train_loader, valid_loader, test_loader = data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y)

    trainer = Trainer(train_loader, valid_loader, test_loader)
    lstm_model = None
    if mode == "study":
        print("--------------------------- STEP 6 STUDYING ---------------------------")
        student = Student(trainer)
        student.study()
        lstm_model = student.train_with_best_params()

    elif mode == "train":
        print("--------------------------- STEP 6 TRAINING ---------------------------")

        # {'hidden_size': 32, 'input_window': 20, 'dropout_rate': 0.2472467993983693,
        #  'learning_rate': 0.009602014523437915}
        # 0.0015953697729855776

        lstm_model = LTSF_LSTM(
            Parser.param_output_window,
            feature_size=Parser.feature_size,
            hidden_size=32,
            dropout=0.247
        ).to(device)

        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0096)
        criterion = nn.MSELoss()

        loss_train, loss_valid, loss_test = trainer.train(Parser.param_epochs, lstm_model, criterion, optimizer)

    print("--------------------------- STEP 7 SHOW RESULT -------------------------")
    if lstm_model is None:
        print("No trained")
    pred = trainer.eval(lstm_model)
    pred = pred[:, :, 0]
    if need_normalize:
        pred = preprocessor.inverse_normalize_test_target(pred)

    real = data_set.test_target
    output = []
    for index, _ in enumerate(real):
        diff_index = index - Parser.param_input_window
        if len(pred) <= diff_index:
            break
        if diff_index < 0:
            output.append(0)
            continue
        else:
            data = real[index - 1] * (1 + (pred[diff_index]/100))
            output.append(data[0])
    output = np.array(output)
    print("Output shape: ", output.shape)
    showTemp(real, np.array(output))
    printt(real, np.array(output), test_y, pred)


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    make_file()
    main()