import argparse

RANDOM_SEED = 42

param_epochs = 50
param_input_window = 50
param_output_window = 7
param_hidden_size = 128
param_learning_rate = 0.001
param_batch_size = 64
param_num_layers = 4
param_dropout = 0.3

def parse():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='study')
    args.add_argument('--epoch', type=int, default=param_epochs,
                      help='epochs, default is 100')
    args.add_argument('--learning_rate', type=float, default=param_learning_rate,
                      help='learning_rate, default is 0.001')
    args.add_argument('--input_window', type=int, default=24,
                      help='input window, default is 72')
    args.add_argument('--output_window', type=int, default=4,
                      help='output window, default is 24')
    args.add_argument('--hidden_size', type=int, default=100,
                      help='hidden size, default is 200')

    return args.parse_args()


def check_params(learning_rate, input_window, output_window, hidden_size, dropout_rate, num_layers):
    if input_window is None or output_window is None or hidden_size is None or learning_rate is None or dropout_rate is None or num_layers is None:
        message = "!! Missing value {}, {} ,{} ,{}, {}".format(input_window, output_window, hidden_size, learning_rate, dropout_rate)
        RuntimeError(message)
    else:
        print("Input window:    ", input_window)
        print("Output window:   ", output_window)
        print("Hidden Size:     ", hidden_size)
        print("Learning rate:   ", learning_rate)