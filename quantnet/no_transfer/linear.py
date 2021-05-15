import numpy as np
import torch
from torch import nn


class NoTransferLinear:
    def __init__(self, x_tasks, model_config):
        self.criterion = self.avg_sharpe_ratio
        self.X_train_tasks = x_tasks
        self.tsteps = model_config["tsteps"]
        self.tasks_tsteps = model_config["tasks_tsteps"]
        self.batch_size = model_config["batch_size"]
        self.seq_len = model_config["seq_len"]
        self.device = model_config["device"]
        self.export_path = model_config["export_path"]
        self.export_label = model_config["export_label"]
        self.opt_lr = model_config["no_transfer_linear"]["opt_lr"]
        self.amsgrad = model_config["no_transfer_linear"]["amsgrad"]
        self.export_weights = model_config["no_transfer_linear"]["export_weights"]

        self.mtl_list = self.X_train_tasks.keys()

        (
            self.sub_mtl_list,
            self.model_lin_dict,
            self.opt_dict,
            self.signal_layer,
            self.losses,
        ) = ({}, {}, {}, {}, {})

        for tk in self.mtl_list:
            (
                self.model_lin_dict[tk],
                self.signal_layer[tk],
                self.opt_dict[tk],
                self.losses[tk],
            ) = ({}, {}, {}, {})

            self.sub_mtl_list[tk] = self.X_train_tasks[tk].keys()

            for sub_tk in self.sub_mtl_list[tk]:
                self.losses[tk][sub_tk] = []
                nin = self.X_train_tasks[tk][sub_tk].shape[1]
                nout = self.X_train_tasks[tk][sub_tk].shape[1]

                self.model_lin_dict[tk][sub_tk] = (
                    nn.Linear(nin, nout).double().to(self.device)
                )
                self.signal_layer[tk][sub_tk] = nn.Tanh().to(self.device)

                self.opt_dict[tk][sub_tk] = torch.optim.Adam(
                    list(self.model_lin_dict[tk][sub_tk].parameters())
                    + list(self.signal_layer[tk][sub_tk].parameters()),
                    lr=self.opt_lr,
                    amsgrad=self.amsgrad,
                )

                print(
                    tk,
                    sub_tk,
                    self.model_lin_dict[tk][sub_tk],
                    self.signal_layer[tk][sub_tk],
                    self.opt_dict[tk][sub_tk],
                )

    def train(self):
        for i in range(self.tsteps):
            for tk in self.mtl_list:
                for sub_tk in self.sub_mtl_list[tk]:
                    start_ids = np.random.permutation(
                        list(
                            range(
                                self.X_train_tasks[tk][sub_tk].size(0)
                                - self.seq_len
                                - 1
                            )
                        )
                    )[: self.batch_size]

                    X_Y_batch = torch.stack(
                        [
                            self.X_train_tasks[tk][sub_tk][i : i + self.seq_len + 1]
                            for i in start_ids
                        ],
                        dim=0,
                    )

                    Y_train = X_Y_batch[:, 1:, :]
                    X_train = X_Y_batch[:, :-1, :]

                    self.opt_dict[tk][sub_tk].zero_grad()

                    preds = self.signal_layer[tk][sub_tk](
                        self.model_lin_dict[tk][sub_tk](X_train)
                    )

                    loss = self.criterion(preds, Y_train)
                    self.losses[tk][sub_tk].append(loss.item())

                    loss.backward()
                    self.opt_dict[tk][sub_tk].step()

            if (i % 100) == 1:
                print(i)

        if self.export_weights:
            for tk in self.mtl_list:
                for sub_tk in self.sub_mtl_list[tk]:
                    torch.save(
                        self.model_lin_dict[tk][sub_tk],
                        self.export_path
                        + tk
                        + "_"
                        + sub_tk
                        + "_"
                        + self.export_label
                        + "_notransferlinear.pt",
                    )

    def predict(self, x_test):
        y_pred = {}

        for tk in self.mtl_list:
            y_pred[tk] = {}

            for sub_tk in self.sub_mtl_list[tk]:
                x_flat = x_test[tk][sub_tk].view(1, -1, x_test[tk][sub_tk].size(1))

                with torch.autograd.no_grad():
                    y_pred[tk][sub_tk] = self.signal_layer[tk][sub_tk](
                        self.model_lin_dict[tk][sub_tk](x_flat[:, :-1])
                    )

        return y_pred

    def avg_sharpe_ratio(self, output, target):
        slip = 0.0005 * 0.00
        bp = 0.0020 * 0.00
        rets = torch.mul(output, target)

        tc = torch.abs(output[:, 1:, :] - output[:, :-1, :]) * (bp + slip)
        tc = torch.cat(
            [
                torch.zeros(output.size(0), 1, output.size(2)).double().to(self.device),
                tc,
            ],
            dim=1,
        )

        rets = rets - tc
        avg_rets = torch.mean(rets)
        vol_rets = torch.std(rets)
        loss = torch.neg(torch.div(avg_rets, vol_rets))

        return loss.mean()
