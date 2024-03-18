from torch.utils.data import Dataset, DataLoader
import numpy as np

class WindowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        # 데이터의 개수
        L = y.shape[0]

        # stride씩 움직이는데 몇번움직임 가능한지
        num_samples = (L - input_window - output_window) // stride + 1

        y_1 = y.transpose((1, 0))[0]
        # input, output
        X = np.zeros([input_window, num_samples, 5])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[:, i] = y[start_x:end_x]

            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[:, i] = y_1[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 5).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1]).transpose((1, 0))
        self.x = X
        self.y = Y
        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len