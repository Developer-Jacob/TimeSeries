import Parser
from trainer import make_trainer
from Const import device
from StockData import StockDataGenerator
import FileManager as fm
from Preprocessor import Preprocessor
from Student import Student
from Util import draw_result, print_result, draw_variance, show_train_log
import Util
import Transformer


def main(execute_mode):
    need_norm = True
    need_diff = True

    print("--------------------------- STEP 1 DATA GENERATOR --------------------")
    generator = StockDataGenerator()
    # data_set = generator.allGenerateData()  # ndarray
    # data_set = generator.half()
    data_set = generator.dummy()
    feature_size = generator.feature_size

    print("--------------------------- STEP 1 PREPARE DATA --------------------")
    preprocessor = Preprocessor(data_set, need_diff=need_diff, need_norm=need_norm, verbose=False)

    print("--------------------------- STEP 2 TRAINING --------------------")
    input_window = None
    output_window = None
    hidden_size = None
    learning_rate = None
    dropout_rate = None
    num_layers = None
    if execute_mode == "study":
        def preprocess(iw, ow):
            return preprocessor.processed(iw, ow)
        student = Student()
        best_params = student.study(feature_size, lambda iw, ow: preprocess(iw, ow))

        input_window = best_params[student.key_input_window]
        output_window = best_params[student.key_output_window]
        hidden_size = best_params[student.key_hidden_size]
        learning_rate = best_params[student.key_learning_rate]
        dropout_rate = best_params[student.key_dropout_rate]
        num_layers = best_params[student.key_num_layers]
    elif execute_mode == "train" or execute_mode == "eval":
        input_window = Parser.param_input_window
        output_window = Parser.param_output_window
        hidden_size = Parser.param_hidden_size
        learning_rate = Parser.param_learning_rate
        dropout_rate = Parser.param_dropout
        num_layers = Parser.param_num_layers

    Parser.check_params(learning_rate, input_window, output_window, hidden_size, dropout_rate, num_layers)

    values = preprocessor.processed(input_window, output_window)
    trainer = make_trainer(values)

    if execute_mode == "train" or execute_mode == "study":
        fm.file_manager.set_params(input_window, output_window, hidden_size, learning_rate, dropout_rate)
        train_loss, valid_loss, test_loss = Util.train_all(trainer, input_window, output_window, feature_size, hidden_size, dropout_rate, learning_rate, num_layers, 4)
        show_train_log(train_loss, valid_loss, test_loss)

    empty_model = Transformer.default_model(
        input_window, output_window, feature_size, hidden_size, dropout_rate, num_layers, 4
    ).to(device)
    trained_model = fm.file_manager.load_model(empty_model)
    pred = trainer.eval(trained_model)

    # print("--------------------------- STEP 3 SHOW --------------------")
    # # pred = pred[:, :, 0]
    #
    # real = data_set.test_target
    # output = []
    # diffed_test_target = None
    #
    # Util.draw_upper_lower(
    #     fm.file_manager.image_path,
    #     real,
    #     upper_bound,
    #     lower_bound,
    #     input_window,
    #     output_window
    # )
    #
    # # draw_result(real, np.array(output), file_manager.image_path)
    # # print_result(file_manager.file_path, real, diffed_test_target, inversed_pred, np.array(output))
    # print_result(fm.file_manager.file_path, 'REAL: ', real)
    # origin_upper = Util.variance_to_origin(real, upper_bound, input_window)
    # origin_lower = Util.variance_to_origin(real, lower_bound, input_window)
    # print_result(fm.file_manager.file_path, 'UPPER', origin_upper)
    # print_result(fm.file_manager.file_path, 'LOWER', origin_lower)

    print("Completed draw, print.")

if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    mode = 'train'
    # mode = 'study'
    # mode = 'eval'
    main(execute_mode=mode)
