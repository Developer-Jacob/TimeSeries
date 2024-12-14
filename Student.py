import optuna
from Util import train_all

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
        input_window = trial.suggest_int(Student.key_input_window, 4, 100, step=4)
        output_window = 1
        dropout_rate = trial.suggest_float(Student.key_dropout_rate, 0.1, 0.5)
        learning_rate = trial.suggest_float(Student.key_learning_rate, 1e-5, 1e-3, log=True)
        num_layers = trial.suggest_int(Student.key_num_layers, 1, 3)
        valid_loss = train_all(None, self.preprocessor, input_window, output_window, hidden_size, dropout_rate, learning_rate, num_layers)
        return valid_loss

    def study(self):
        study = optuna.create_study()
        study.optimize(lambda trial: self.objective(trial), n_trials=500)
        print(study.best_params)
        print(study.best_value)
        return study.best_params
