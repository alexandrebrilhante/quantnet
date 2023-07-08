import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class BuyAndHold:
    def __init__(self, x_tasks, model_config):
        self.X_train_tasks = x_tasks
        self.export_path = model_config["export_path"]
        self.export_label = model_config["export_label"]

        self.mtl_list = self.X_train_tasks.keys()
        self.sub_mtl_list = {}
        self.signal = {}

        for tk in self.mtl_list:
            self.signal[tk] = {}
            self.sub_mtl_list[tk] = self.X_train_tasks[tk].keys()

            for sub_tk in self.sub_mtl_list[tk]:
                self.signal[tk][sub_tk] = LinearRegression()

    def train(self):
        for tk in self.mtl_list:
            for sub_tk in self.sub_mtl_list[tk]:
                X_train = self.X_train_tasks[tk][sub_tk][:-1, :]
                Y_train = self.X_train_tasks[tk][sub_tk][1:, :]

                self.signal[tk][sub_tk].fit(X_train, Y_train)
                self.signal[tk][sub_tk].intercept_ = 1.0
                self.signal[tk][sub_tk].coef_ = self.signal[tk][sub_tk].coef_ * 0.0

                print(tk, sub_tk)

    def predict(self, x_test):
        y_pred = {}

        for tk in self.mtl_list:
            y_pred[tk] = {}

            for sub_tk in self.sub_mtl_list[tk]:
                y_pred[tk][sub_tk] = self.signal[tk][sub_tk].predict(x_test[tk][sub_tk])

        return y_pred


class RiskParity:
    def __init__(self, x_tasks, model_config):
        self.X_train_tasks = x_tasks
        self.export_path = model_config["export_path"]
        self.export_label = model_config["export_label"]
        self.window = model_config["risk_parity"]["window"]

        self.mtl_list = self.X_train_tasks.keys()
        self.sub_mtl_list = {}
        self.signal = {}

        for tk in self.mtl_list:
            self.signal[tk] = {}
            self.sub_mtl_list[tk] = self.X_train_tasks[tk].keys()

            for sub_tk in self.sub_mtl_list[tk]:
                self.signal[tk][sub_tk] = LinearRegression()

    def train(self):
        for tk in self.mtl_list:
            for sub_tk in self.sub_mtl_list[tk]:
                X_train = self.X_train_tasks[tk][sub_tk][self.window : -1, :]
                Y_train = self.X_train_tasks[tk][sub_tk][self.window + 1 :, :]

                self.signal[tk][sub_tk].fit(X_train, Y_train)
                self.signal[tk][sub_tk].intercept_ = 1.0
                self.signal[tk][sub_tk].coef_ = self.signal[tk][sub_tk].coef_ * 0.0

                print(tk, sub_tk)

    def predict(self, x_test):
        y_pred = {}

        for tk in self.mtl_list:
            y_pred[tk] = {}
            for sub_tk in self.sub_mtl_list[tk]:
                x = pd.DataFrame(
                    np.concatenate(
                        [self.X_train_tasks[tk][sub_tk], x_test[tk][sub_tk]], axis=0
                    )
                )

                risk = (
                    x.rolling(window=self.window)
                    .std()
                    .values[-x_test[tk][sub_tk].shape[0] :, :]
                )

                y_pred[tk][sub_tk] = (1.0 / risk) / np.repeat(
                    np.sum((1.0 / risk), axis=1).reshape(-1, 1), risk.shape[1], axis=1
                )

        return y_pred


class TimeSeriesMomentum:
    def __init__(self, x_tasks, model_config):
        self.X_train_tasks = x_tasks
        self.export_path = model_config["export_path"]
        self.export_label = model_config["export_label"]
        self.window = model_config["ts_mom"]["window"]

        self.mtl_list = self.X_train_tasks.keys()
        self.sub_mtl_list = {}
        self.signal = {}

        for tk in self.mtl_list:
            self.signal[tk] = {}
            self.sub_mtl_list[tk] = self.X_train_tasks[tk].keys()

            for sub_tk in self.sub_mtl_list[tk]:
                self.signal[tk][sub_tk] = LinearRegression()

    def train(self):
        for tk in self.mtl_list:
            for sub_tk in self.sub_mtl_list[tk]:
                X_train = pd.DataFrame(self.X_train_tasks[tk][sub_tk][:-1, :])
                Y_train = pd.DataFrame(self.X_train_tasks[tk][sub_tk][1:, :])

                self.signal[tk][sub_tk].fit(X_train, Y_train)
                self.signal[tk][sub_tk].intercept_ = 1.0
                self.signal[tk][sub_tk].coef_ = self.signal[tk][sub_tk].coef_ * 0.0

                print(tk, sub_tk)

    def predict(self, x_test):
        y_pred = {}

        for tk in self.mtl_list:
            y_pred[tk] = {}

            for sub_tk in self.sub_mtl_list[tk]:
                x = pd.DataFrame(
                    np.concatenate(
                        [self.X_train_tasks[tk][sub_tk], x_test[tk][sub_tk]], axis=0
                    )
                )

                y_pred[tk][sub_tk] = (
                    -x.rolling(window=self.window)
                    .mean()
                    .values[-x_test[tk][sub_tk].shape[0] :, :]
                )

        return y_pred


class CrossSectionalMomentum:
    def __init__(self, x_tasks, model_config):
        self.X_train_tasks = x_tasks
        self.export_path = model_config["export_path"]
        self.export_label = model_config["export_label"]
        self.window = model_config["csec_mom"]["window"]
        self.fraction = model_config["csec_mom"]["fraction"]

        self.mtl_list = self.X_train_tasks.keys()
        self.sub_mtl_list = {}
        self.signal = {}

        for tk in self.mtl_list:
            self.signal[tk] = {}
            self.sub_mtl_list[tk] = self.X_train_tasks[tk].keys()

            for sub_tk in self.sub_mtl_list[tk]:
                self.signal[tk][sub_tk] = LinearRegression()

    def train(self):
        for tk in self.mtl_list:
            for sub_tk in self.sub_mtl_list[tk]:
                X_train = pd.DataFrame(self.X_train_tasks[tk][sub_tk][:-1, :])
                Y_train = pd.DataFrame(self.X_train_tasks[tk][sub_tk][1:, :])

                self.signal[tk][sub_tk].fit(X_train, Y_train)
                self.signal[tk][sub_tk].intercept_ = 1.0
                self.signal[tk][sub_tk].coef_ = self.signal[tk][sub_tk].coef_ * 0.0

                print(tk, sub_tk)

    def predict(self, x_test):
        y_pred = {}

        for tk in self.mtl_list:
            y_pred[tk] = {}

            for sub_tk in self.sub_mtl_list[tk]:
                x = pd.DataFrame(
                    np.concatenate(
                        [self.X_train_tasks[tk][sub_tk], x_test[tk][sub_tk]], axis=0
                    )
                )

                signal = (
                    x.rolling(window=self.window)
                    .mean()
                    .values[-x_test[tk][sub_tk].shape[0] :, :]
                )

                bottom = (
                    pd.DataFrame(signal).rank(axis=1) / signal.shape[1]
                ).values < self.fraction

                top = (pd.DataFrame(signal).rank(axis=1) / signal.shape[1]).values > (
                    1 - self.fraction
                )

                y_pred[tk][sub_tk] = np.multiply(-signal, (bottom + top))

        return y_pred
