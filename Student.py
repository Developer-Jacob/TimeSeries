import optuna
from model_lstm import LTSF_LSTM
from Const import device
import torch
import torch.nn as nn
import Parser


#def _make_trainer(_preprocessor, _input_windows, _output_windows):
#def _make_model(_output_window, _hidden_size, _drop_out, _learning_rate):

def objective(trial, epochs, preprocessor, function_make_trainer, function_make_model):
    hidden_size = trial.suggest_int(Student.key_hidden_size, 32, 256, step=32)
    input_window = trial.suggest_int(Student.key_input_window, 4, 100, step=4)
    output_window = 1
    dropout_rate = trial.suggest_float(Student.key_dropout_rate, 0.1, 0.5)
    learning_rate = trial.suggest_float(Student.key_learning_rate, 1e-5, 1e-1, log=True)

    trainer = function_make_trainer(preprocessor, input_window, output_window)
    model, optimizer, criterion = function_make_model(output_window, hidden_size, dropout_rate, learning_rate)

    loss_train, loss_valid, loss_test = trainer.train(epochs, model, criterion, optimizer)

    return loss_valid


class Student:
    key_hidden_size = 'hidden_size'
    key_dropout_rate = 'dropout_rate'
    key_learning_rate = 'learning_rate'
    key_input_window = 'input_window'

    def __init__(self, epochs, preprocessor, function_make_trainer, function_make_model):
        self.epochs = epochs
        self.function_make_trainer = function_make_trainer
        self.function_make_model = function_make_model
        self.preprocessor = preprocessor
        self.best_prams = None

    def best_input_window(self):
        return self.best_prams[Student.key_input_window]

    def study(self):
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, self.epochs,  self.preprocessor, self.function_make_trainer, self.function_make_model), n_trials=500)
        print(study.best_params)
        print(study.best_value)
        self.best_prams = study.best_params

    def train_with_best_params(self):
        input_windows = self.best_prams[Student.key_input_window]
        hidden_size = self.best_prams[Student.key_hidden_size]
        dropout_rate = self.best_prams[Student.key_dropout_rate]
        learning_rate = self.best_prams[Student.key_learning_rate]

        trainer = self.function_make_trainer(self.preprocessor, input_windows, 1)
        lstm_model = LTSF_LSTM(
            Parser.param_output_window,
            feature_size=Parser.feature_size,
            hidden_size=hidden_size,
            dropout=dropout_rate
        ).to(device)

        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        loss_train, loss_valid, loss_test = trainer.train(self.epochs, lstm_model, criterion, optimizer)
        return lstm_model, trainer

