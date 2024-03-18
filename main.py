import matplotlib.pyplot as plt
import gc
from sklearn.preprocessing import MinMaxScaler
from window_data import *
from trainer import *
from row_data import *
import torch

def device_checker():
    print(torch.cuda.is_available())

def show(real, result):
    max = len(real)

    plt.figure(figsize=(20, 5))
    plt.plot(range(0, max), real, label="real")
    plt.plot(range(max - predict_count, max), result, label="predict")
    plt.legend()
    plt.show()
    print('completed show')

def start():
    device_checker()

    data = RowData(input_window, output_window)

    train_dataset = WindowDataset(data.train_data, input_window=input_window, output_window=output_window, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    del train_dataset
    gc.collect()

    trainer = Trainer(input_window, output_window, PATH)
    trainer.train(epoch, train_loader)

    input = torch.from_numpy(data.test_data)
    result = trainer.evaluate(input)

    result = data.scaler.inverse_transform(result.reshape(-1, 1))[:, 0]



    real = data.all_data
    show(real, result)

# static
PATH = './time_serial_100_200.pt'
NAME = 'time_serial.pt'
predict_count = 100
input_window = predict_count * 2
output_window = predict_count
epoch = 1000
batch_size = 64

if __name__ == '__main__':
    start()

