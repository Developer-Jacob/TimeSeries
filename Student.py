import optuna
from model_lstm import LTSF_LSTM
import Parser
from Const import device
import torch
import torch.nn as nn


class Student:
    key_hidden_size = 'hidden_size'
    key_dropout_rate = 'dropout_rate'
    key_learning_rate = 'learning_rate'

    def __init__(self, trainer):
        self.trainer = trainer
        self.best_prams = None

    def objective(self, trial, trainer):
        hidden_size = trial.suggest_int(Student.key_hidden_size, 32, 256, step=32)
        # input_window = trial.suggest_int('input_window', 10, 100, step=5)
        dropout_rate = trial.suggest_float(Student.key_dropout_rate, 0.1, 0.5)
        learning_rate = trial.suggest_float(Student.key_learning_rate, 1e-5, 1e-1, log=True)

        lstm_model = LTSF_LSTM(
            Parser.param_output_window,
            feature_size=Parser.feature_size,
            hidden_size=hidden_size,
            dropout=dropout_rate
        ).to(device)

        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        loss_train, loss_valid, loss_test = trainer.train(Parser.param_epochs, lstm_model, criterion, optimizer)

        return loss_valid

    def study(self):
        study = optuna.create_study()
        study.optimize(lambda trial: self.objective(trial, self.trainer), n_trials=500)
        print(study.best_params)
        print(study.best_value)
        self.best_prams = study.best_params

    def train_with_best_params(self):
        hidden_size = self.best_prams[Student.key_hidden_size]
        dropout_rate = self.best_prams[Student.key_dropout_rate]
        learning_rate = self.best_prams[Student.key_learning_rate]

        lstm_model = LTSF_LSTM(
            Parser.param_output_window,
            feature_size=Parser.feature_size,
            hidden_size=hidden_size,
            dropout=dropout_rate
        ).to(device)

        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        loss_train, loss_valid, loss_test = self.trainer.train(Parser.param_epochs, lstm_model, criterion, optimizer)
        return lstm_model

