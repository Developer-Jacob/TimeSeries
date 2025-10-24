

class Config:
    def __init__(self, total_len):
        self.seq_len = int(total_len * 0.8) # input size
        self.window_shift = 1
        self.num_features = num_features  # Features
        self.d_model = 20  # Convolution Embedding dimension AFTER RESHAPING
        self.top_k = 3  # FFT frequency
        self.d_ff = 20  # Convolution Output layer dimension AFTER RESHAPING
        self.num_kernels = 6  # inception block Num of different grid cells used / If using dcvn set it to 3
        self.dropout = 0.1933493411095017  # Dropout rate
        self.e_layers = 1  # num Timeblock
        self.label_len = num_features  # Features
        self.target_col = target_name  # Name of target column
        self.cnn_type = 'inceptionv1'  # dcvn (KERNEL = 3), inceptionv1, inceptionv2, res_dcvn, res_inceptionv1, res_inceptionv2
        self.pred_len = 20  # Prediction length
        self.c_out = 1  # Output feature
        self.eval_range = 0
        # self.seq_range = np.arange(45, 317)
        self.seq_range = slice(None)
        self.scheduler_config = SchedulerConfig()
        self.scheduler_name = 'CosineAnnealingWarmRestarts'  # 'CosineAnnealingWarmRestarts', 'StepLR', 'ExponentialLR', 'OneCycleLR', 'CyclicLR'
        self.scheduler_update_type = 'epoch'  # epoch, batch