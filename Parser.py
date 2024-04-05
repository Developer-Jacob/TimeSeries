import argparse

# num_epochs = 1
# learning_rate = 0.0001
# input_window = 128
# output_window = 4
# hidden_size = 200
# num_layers = 1


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
