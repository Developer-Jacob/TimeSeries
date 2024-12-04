import matplotlib.pyplot as plt
from datetime import datetime
import Parser

def showTemp(real, result, section=100):
    fig = plt.figure(figsize=(20, 5))

    start = len(real) - section
    if start < 0:
        start = 0
    end = len(real) - 1
    plt.plot(range(start, end), result[start:end], 'b.-')
    plt.plot(range(start, end), real[start:end], 'r.-')
    plt.savefig(path() + '/result.png')

def path():
    date_str = datetime.today().strftime("%Y%m%d")
    directory = 'EP{}_IW{}_OW{}_HS{}_LR{}'.format(
        Parser.param_epochs,
        Parser.param_input_window,
        Parser.param_output_window,
        Parser.param_hidden_size,
        Parser.param_learning_rate
    )
    return './Model/{}/{}'.format(date_str, directory)