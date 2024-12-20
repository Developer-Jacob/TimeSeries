import optuna
from Util import train_all, show_train_log
from trainer import make_trainer
from EarlyStopping import EarlyStopping
import numpy as np

class Student:
    key_hidden_size = 'hidden_size'
    key_dropout_rate = 'dropout_rate'
    key_learning_rate = 'learning_rate'
    key_input_window = 'input_window'
    key_num_layers = 'num_layers'
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def objective(self, trial):
        hidden_size = trial.suggest_int(Student.key_hidden_size, 32, 128, step=32)
        input_window = trial.suggest_int(Student.key_input_window, 50, 100, step=10)
        output_window = 1
        dropout_rate = trial.suggest_float(Student.key_dropout_rate, 0.3, 0.5, step=0.05)
        learning_rate = trial.suggest_float(Student.key_learning_rate, 0.0001, 0.001, log=True)
        num_layers = trial.suggest_int(Student.key_num_layers, 2, 4)
        print("--------------------------- Study --------------------")
        print("Spec input_window {}, output_window {}, hidden_size {}, dropout_rate {}, learning_rate {}, num_layers {}"
              .format(input_window, output_window, hidden_size, dropout_rate, learning_rate, num_layers))
        values = self.preprocessor.processed(input_window, output_window)
        trainer = make_trainer(values)
        early_stopping = EarlyStopping(None, patience=10, verbose=True)
        train_loss, valid_loss, test_loss = train_all(trainer, early_stopping, input_window, output_window, self.preprocessor.feature_size, hidden_size, dropout_rate, learning_rate, num_layers)
        show_train_log(train_loss, valid_loss, test_loss)
        return np.mean(valid_loss)


    def study(self):
        study = optuna.create_study()
        study.optimize(lambda trial: self.objective(trial), n_trials=10)
        print(study.best_params)
        print(study.best_value)
        return study.best_params
