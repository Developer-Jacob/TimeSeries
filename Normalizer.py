from sklearn.preprocessing import MinMaxScaler


class Normalizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.array_scaler = []

    def fit(self, fit_data, transform_data):
        for item_index in range(len(fit_data)):
            f_data = fit_data[item_index]
            t_data = transform_data[item_index]
            scaler = MinMaxScaler()
            for feature_index in range(len(f_data)):
                if feature_index == 0:
                    scaler.fit_transform(f_data[feature_index])
                    scaler.transform(t_data)
                    self.array_scaler.append(scaler)
                else:
                    else_scaler = MinMaxScaler()
                    else_scaler.fit_transform(f_data[feature_index])

        # return
    def normalize(self, data):
        print("")