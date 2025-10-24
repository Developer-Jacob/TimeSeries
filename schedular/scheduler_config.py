class SchedulerConfig:
    def __init__(self):
        # STARTS from lr goes down to eta_min in T_0
        self.CosineAnnealingWarmRestarts = {'T_0': 10, 'T_mult': 1, 'eta_min': 0.0005}
        self.StepLR = {'step_size': 10, 'gamma': 0.1}
        self.ExponentialLR = {'gamma': 0.95}

        # steps_per_epoch should be set to number of batches , epochs should be total number of epochs
        # Starts from low lr to max lr
        self.OneCycleLR = {'max_lr': 0.01, 'steps_per_epoch': 10, 'epochs': 20}

        self.CyclicLR = {'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5, 'step_size_down': 5, 'mode': 'triangular'}

    def get_params(self, scheduler_name):
        return getattr(self, scheduler_name, None)