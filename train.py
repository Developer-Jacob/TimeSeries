from StockData import StockData
from StockDataGenerator import StockDataGenerator
from Preprocessor import Preprocessor
import Parser
from trainer import make_trainer
import FileManager as fm
import Util
import Transformer
from Util import show_train_log
from Const import device


def train():
    print("--------------------------- STEP 1 DATA GENERATOR --------------------")
    generator = StockDataGenerator()
    data_set = generator.mini()  # ndarray
    feature_size = generator.feature_size

    print("--------------------------- STEP 2 PREPARE DATA --------------------")
    input_window = Parser.param_input_window
    output_window = Parser.param_output_window
    hidden_size = Parser.param_hidden_size
    learning_rate = Parser.param_learning_rate
    dropout_rate = Parser.param_dropout
    num_layers = Parser.param_num_layers
    num_heads = Parser.param_num_head
    Parser.check_params(learning_rate, input_window, output_window, hidden_size, dropout_rate, num_layers, num_heads)

    print("--------------------------- STEP 3 PREPROCESSING --------------------")
    preprocessor = Preprocessor(data_set, need_diff=Parser.diff, need_norm=Parser.norm, verbose=False)
    values = preprocessor.processed(input_window, output_window)
    trainer = make_trainer(values)

    print("--------------------------- STEP 4 TRAINING --------------------")
    fm.file_manager.set_params(input_window, output_window, hidden_size, learning_rate, dropout_rate)
    train_loss, valid_loss, test_loss = Util.train_all(trainer, input_window, output_window, feature_size, hidden_size, dropout_rate, learning_rate, num_layers, num_heads)
    show_train_log("Show train log.", train_loss, valid_loss, test_loss)

    print("--------------------------- STEP 4 PREDICTION --------------------")
    empty_model = Transformer.default_model(input_window, output_window, feature_size, hidden_size, dropout_rate, num_layers, num_heads)
    empty_model = empty_model.to(device)

    trained_model = fm.file_manager.load_model(empty_model)
    predictions = trainer.eval(trained_model)


    print('--------------------------- STEP 5 SHOW RESULT --------------------')
    real = data_set.test_target

    diffed = preprocessor.diffed()[5]
    diffed_pred = [preprocessor.inverse_normalize_test_target(pred) for pred in predictions]
    Util.draw_quantiles(diffed[-len(diffed_pred[0]):], diffed_pred)

if __name__ == '__main__':
    train()