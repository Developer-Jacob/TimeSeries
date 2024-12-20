import optuna
from Util import train_all
from trainer import make_trainer
from EarlyStopping import EarlyStopping


class Student:
    key_hidden_size = 'hidden_size'
    key_dropout_rate = 'dropout_rate'
    key_learning_rate = 'learning_rate'
    key_input_window = 'input_window'
    key_num_layers = 'num_layers'
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    {'hidden_size': 64, 'input_window': 90, 'dropout_rate': 0.310854294481087, 'learning_rate': 2.6351132090717395e-05,
     'num_layers': 2}.Best is trial
    99
    with value: 0.4027290573486915.
    def objective(self, trial):
        hidden_size = trial.suggest_int(Student.key_hidden_size, 32, 128, step=32)
        input_window = trial.suggest_int(Student.key_input_window, 50, 100, step=10)
        output_window = 1
        dropout_rate = trial.suggest_float(Student.key_dropout_rate, 0.3, 0.5)
        learning_rate = trial.suggest_float(Student.key_learning_rate, 1e-5, 1e-3, log=True)
        num_layers = trial.suggest_int(Student.key_num_layers, 2, 4)
        print("--------------------------- Study --------------------")
        print("Spec input_window {}, output_window {}, hidden_size {}, dropout_rate {}, learning_rate {}, num_layers {}"
              .format(input_window, output_window, hidden_size, dropout_rate, learning_rate, num_layers))
        values = self.preprocessor.processed(input_window, output_window)
        trainer = make_trainer(values)
        early_stopping = EarlyStopping(None, patience=20, verbose=True)
        valid_loss = train_all(trainer, early_stopping, input_window, output_window, self.preprocessor.feature_size, hidden_size, dropout_rate, learning_rate, num_layers)
        return valid_loss

    def study(self):
        study = optuna.create_study()
        study.optimize(lambda trial: self.objective(trial), n_trials=100)
        print(study.best_params)
        print(study.best_value)
        return study.best_params
