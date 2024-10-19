import argparse

RANDOM_SEED = 42
param_is_debug = False
param_epochs = 200
param_encoder_epochs = 10
param_input_window = 4
param_output_window = 1
param_hidden_size = 256
param_learning_rate = 0.001
param_batch_size = 64
param_is_train_mode = True

def parse():
    args = argparse.ArgumentParser()
    args.add_argument('--start', type=str, default='eval',
                      help='all, train, eval')
    args.add_argument('--epoch', type=int, default=200,
                      help='epochs, default is 100')
    args.add_argument('--learning_rate', type=float, default='0.001',
                      help='learning_rate, default is 0.001')
    args.add_argument('--input_window', type=int, default=24,
                      help='input window, default is 72')
    args.add_argument('--output_window', type=int, default=4,
                      help='output window, default is 24')
    args.add_argument('--hidden_size', type=int, default=100,
                      help='hidden size, default is 200')

    return args.parse_args()


config = parse()
