import optuna
from model_lstm import LTSF_LSTM
import Parser
from Const import device
import torch
import torch.nn as nn


def objective(trial, trainer):
    hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
    input_window = trial.suggest_int('input_window', 10, 100, step=5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    lstm_model = LTSF_LSTM(
        input_window,
        Parser.param_output_window,
        feature_size=Parser.feature_size,
        hidden_size=hidden_size,
        dropout=dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss_train, loss_valid, loss_test = trainer.train(Parser.param_epochs, lstm_model, criterion, optimizer)

    return loss_valid


def study(trainer):
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, trainer), n_trials=500)
    print(study.best_params)
    print(study.best_value)