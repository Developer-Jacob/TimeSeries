from datetime import datetime
import os
import torch


class FileManager:
    def __init__(self):
        self.model_path = None
        self.file_path = None
        self.image_path = None

    def save_model(self, model):
        torch.save({'state_dict': model.state_dict()}, self.model_path)

    def load_model(self, model):
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def set_params(self, input_window, output_window, hidden_size, learning_rate, drop_out):
        self.input_window = input_window
        self.output_window = output_window
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.drop_out = drop_out
        self._make_directory(
            input_window,
            output_window,
            hidden_size,
            learning_rate,
            drop_out
        )
        self.file_path = self.directory_path + "/" + 'result.txt'
        self.model_path = self.directory_path + "/" + 'model.pth'
        self.image_path = self.directory_path + "/" + 'result.png'
        self._save_text()

    def _make_directory(self, input_window, output_window, hidden_size, learning_rate, drop_out):
        date_str = datetime.today().strftime("%Y%m%d")
        os.makedirs(date_str, exist_ok=True)
        directory = 'IW{}_OW{}_HS{}_LR{:.4f}_DO{:.4f}'.format(
            input_window,
            output_window,
            hidden_size,
            learning_rate,
            drop_out
        )
        self.directory_path = './Model/{}/{}'.format(date_str, directory)
        os.makedirs(self.directory_path, exist_ok=True)

    def _save_text(self):
        f = open(self.file_path, 'w+')
        f.write('\nInputWindow {}'.format(self.input_window))
        f.write('\nOutputWindow {}'.format(self.output_window))
        f.write('\nHiddenSize {}'.format(self.hidden_size))
        f.write('\nLearningRate {}'.format(self.learning_rate))
        f.write('\nDropout {}'.format(self.drop_out))
        f.close()