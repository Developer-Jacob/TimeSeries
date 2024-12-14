import Parser
from trainer import make_trainer
from Const import device
from StockData import StockDataGenerator
from FileManager import FileManager
from Preprocessor import Preprocessor
from Student import Student
from Util import draw_result, print_result, draw_variance
import Util
import numpy as np
from model_lstm import lstm_model
from Differ import restore_data, restore_target


def main():
    file_manager = FileManager()
    print("Device: ", device)
    need_norm = True
    need_diff = True
    print("--------------------------- STEP 1 DATA GENERATOR --------------------")
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()  # ndarray
    # data_set = generator.dummy()

    mode = "train"

    epochs = Parser.param_epochs
    preprocessor = Preprocessor(data_set, generator.feature_size, need_diff=need_diff, need_norm=need_norm, verbose=False)
    print("--------------------------- STEP 2 TRAINING --------------------")

    input_window = None
    output_window = 1
    hidden_size = None
    learning_rate = None
    dropout = None
    num_layers = None
    if mode == "study":
        student = Student(preprocessor)
        best_params = student.study()

        input_window = best_params[student.key_input_window]
        hidden_size = best_params[student.key_hidden_size]
        learning_rate = best_params[student.key_learning_rate]
        dropout = best_params[student.key_dropout_rate]
        num_layers = best_params[student.key_num_layers]
    elif mode == "train" or mode == "eval":
        # input_window = Parser.param_input_window
        input_window = 88
        output_window = Parser.param_output_window
        # hidden_size = Parser.param_hidden_size
        hidden_size = 64
        # learning_rate = Parser.param_learning_rate
        learning_rate = 0.0001
        dropout = 0.4

        num_layers = 2

    if input_window is None or output_window is None or hidden_size is None or learning_rate is None or dropout is None or num_layers is None:
        print("!! Missing value", input_window, output_window, hidden_size, learning_rate, dropout)
        return

    is_eval_mode = mode == "eval"
    valid_loss = Util.train_all(file_manager, preprocessor, input_window, output_window, hidden_size, dropout, learning_rate, num_layers, is_eval_mode)
    values = preprocessor.processed(input_window, output_window)
    trainer = make_trainer(file_manager, values)

    empty_model = lstm_model(
        output_window=output_window,
        feature_size=preprocessor.feature_size,
        hidden_size=hidden_size,
        dropout_rate=dropout,
        num_layers=num_layers
    ).to(device)
    trained_model = file_manager.load_model(empty_model)
    pred = trainer.eval(trained_model)

    print("--------------------------- STEP 3 SHOW --------------------")
    pred = pred[:, :, 0]

    if need_norm:
        inversed_pred = preprocessor.inverse_normalize_test_target(pred)
    else:
        inversed_pred = pred

    real = data_set.test_target
    output = []
    diffed_test_target = None
    if need_diff:
        for index, _ in enumerate(real):
            diff_index = index - input_window - 1
            if len(inversed_pred) <= diff_index:
                break
            if diff_index < 0:
                output.append(0)
                continue
            else:
                data = real[index - 1] * (1 + (inversed_pred[diff_index]/100))
                output.append(data[0])
        output = np.array(output)
        diffed_test_target = preprocessor.diffed()[5]
        draw_variance(diffed_test_target, inversed_pred, file_manager.variance_image_path)
    else:
        output = inversed_pred
        for i in range(0, input_window):
            output = np.insert(output, 0, 0)

    draw_result(real, np.array(output), file_manager.image_path)
    print_result(file_manager.file_path, real, diffed_test_target, inversed_pred, np.array(output))
    print("Completed draw, print.")

if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    main()