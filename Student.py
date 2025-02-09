import optuna
from Util import train_all, show_train_log
from trainer import make_trainer
import numpy as np

class Student:
    key_hidden_size = 'hidden_size'
    key_dropout_rate = 'dropout_rate'
    key_learning_rate = 'learning_rate'
    key_input_window = 'input_window'
    key_output_window = 'output_window'
    key_num_layers = 'num_layers'

    def objective(self, trial, feature_size, preprocess_block):
        hidden_size = trial.suggest_int(Student.key_hidden_size, 64, 256, step=32)
        input_window = trial.suggest_int(Student.key_input_window, 50, 100, step=10)
        output_window = trial.suggest_int(Student.key_output_window, 5, 20, step=1)
        dropout_rate = trial.suggest_float(Student.key_dropout_rate, 0.2, 0.4, step=0.05)
        learning_rate = trial.suggest_float(Student.key_learning_rate, 0.0005, 0.001, log=True)
        num_layers = trial.suggest_int(Student.key_num_layers, 2, 4)
        print("--------------------------- Study --------------------")
        title = 'IW{}_OW{}_HS{}_LR{:.4f}_DO{:.4f}'.format(
            input_window,
            output_window,
            hidden_size,
            learning_rate,
            dropout_rate
        )

        values = preprocess_block(input_window, output_window)

        print("Spec input_window {}, output_window {}, hidden_size {}, dropout_rate {}, learning_rate {}, num_layers {}"
              .format(input_window, output_window, hidden_size, dropout_rate, learning_rate, num_layers))
        trainer = make_trainer(values)
        trainer.save_mode = False
        train_loss, valid_loss, test_loss = train_all(trainer, input_window, output_window, feature_size, hidden_size, dropout_rate, learning_rate, num_layers, 4)
        show_train_log(title, train_loss, valid_loss, test_loss)
        return np.mean(valid_loss)

    def study(self, feature_size, preprocess_block):
        study = optuna.create_study()
        study.optimize(lambda trial: self.objective(trial, feature_size, preprocess_block), n_trials=50)
        print(study.best_params)
        print(study.best_value)
        return study.best_params
