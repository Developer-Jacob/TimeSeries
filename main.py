import Parser
from trainer import make_trainer
from Const import device
from StockData import StockDataGenerator
from FileManager import FileManager
from Preprocessor import Preprocessor
from Student import Student
from Util import draw_result, print_result, draw_variance, show_train_log
import Util
import numpy as np
from model_lstm import lstm_model
from Transformer import TimeSeriesTransformer
from EarlyStopping import EarlyStopping

def main():
    file_manager = FileManager()
    print("Device: ", device)
    need_norm = True
    need_diff = True
    print("--------------------------- STEP 1 DATA GENERATOR --------------------")
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()  # ndarray
    # data_set = generator.dummy()

    mode = "study"

    epochs = Parser.param_epochs
    preprocessor = Preprocessor(data_set, generator.feature_size, need_diff=need_diff, need_norm=need_norm, verbose=False)
    print("--------------------------- STEP 2 TRAINING --------------------")

    input_window = None
    output_window = 1
    hidden_size = None
    learning_rate = None
    dropout_rate = None
    num_layers = None
    if mode == "study":
        student = Student(preprocessor)
        best_params = student.study()

        input_window = best_params[student.key_input_window]
        hidden_size = best_params[student.key_hidden_size]
        learning_rate = best_params[student.key_learning_rate]
        dropout_rate = best_params[student.key_dropout_rate]
        num_layers = best_params[student.key_num_layers]
    elif mode == "train" or mode == "eval":

        # input_window = Parser.param_input_window
        input_window = 90
        output_window = Parser.param_output_window
        # hidden_size = Parser.param_hidden_size
        hidden_size = 256
        # learning_rate = Parser.param_learning_rate
        learning_rate = 0.0001
        dropout_rate = 0.3

        num_layers = 3

    if input_window is None or output_window is None or hidden_size is None or learning_rate is None or dropout_rate is None or num_layers is None:
        print("!! Missing value", input_window, output_window, hidden_size, learning_rate, dropout_rate)
        return
    Parser.print_params(Parser.param_epochs, learning_rate, input_window, output_window, hidden_size, Parser.param_batch_size)
    values = preprocessor.processed(input_window, output_window)
    trainer = make_trainer(values)

    if mode == "train" or mode == "study":
        file_manager.set_params(input_window, output_window, hidden_size, learning_rate, dropout_rate)
        early_stopping = EarlyStopping(file_manager, patience=10, verbose=True)
        train_loss, valid_loss, test_loss = Util.train_all(trainer, early_stopping, input_window, output_window, preprocessor.feature_size, hidden_size, dropout_rate, learning_rate, num_layers)
        show_train_log(train_loss, valid_loss, test_loss)

    empty_model = TimeSeriesTransformer(
        input_dim=preprocessor.feature_size,
        d_model=hidden_size,
        n_heads=4,
        num_layers=num_layers,
        seq_len=input_window,
        output_dim=output_window,
        dropout_rate=dropout_rate
    ).to(device)
    trained_model = file_manager.load_model(empty_model)
    pred = trainer.eval(trained_model)

    print("--------------------------- STEP 3 SHOW --------------------")
    # pred = pred[:, :, 0]
    pred = pred.squeeze()
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
                output.append(data.squeeze())
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
