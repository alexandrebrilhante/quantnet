import numpy as np
import torch
from torch import nn


class GlobalLstmLinear:
    def __init__(self, x_tasks, model_config):
        self.criterion = self.avg_sharpe_ratio
        self.X_train_tasks = x_tasks
        self.t_steps = model_config["t_steps"]
        self.tasks_t_steps = model_config["tasks_t_steps"]
        self.batch_size = model_config["batch_size"]
        self.seq_len = model_config["seq_len"]
        self.device = model_config["device"]
        self.export_path = model_config["export_path"]
        self.export_label = model_config["export_label"]
        self.opt_lr = model_config["global_lstm_linear"]["opt_lr"]
        self.amsgrad = model_config["global_lstm_linear"]["amsgrad"]
        self.export_weights = model_config["global_lstm_linear"]["export_model"]
        self.in_transfer_dim = model_config["global_lstm_linear"]["in_transfer_dim"]
        self.out_transfer_dim = model_config["global_lstm_linear"]["out_transfer_dim"]
        self.transfer_layers = model_config["global_lstm_linear"]["n_layers"]
        self.dropout = model_config["global_lstm_linear"]["drop_rate"]

        self.mtl_list = self.X_train_tasks.keys()

        self.sub_mtl_list = {}
        (
            self.model_in_dict,
            self.model_out_dict,
            self.opt_dict,
            self.signal_layer,
            self.losses,
        ) = ({}, {}, {}, {}, {})

        self.global_transfer_lstm = (
            nn.LSTM(
                self.in_transfer_dim,
                self.out_transfer_dim,
                self.transfer_layers,
                batch_first=True,
                dropout=self.dropout,
            )
            .double()
            .to(self.device)
        )

        for tk in self.mtl_list:
            (
                self.model_in_dict[tk],
                self.model_out_dict[tk],
                self.signal_layer[tk],
                self.opt_dict[tk],
                self.losses[tk],
            ) = ({}, {}, {}, {}, {})

            self.sub_mtl_list[tk] = self.X_train_tasks[tk].keys()

            for sub_tk in self.sub_mtl_list[tk]:
                self.losses[tk][sub_tk] = []
                nin = self.X_train_tasks[tk][sub_tk].shape[1]
                nout = self.X_train_tasks[tk][sub_tk].shape[1]

                self.model_in_dict[tk][sub_tk] = (
                    nn.Linear(nin, self.in_transfer_dim).double().to(self.device)
                )
                self.model_out_dict[tk][sub_tk] = (
                    nn.Linear(self.out_transfer_dim, nout).double().to(self.device)
                )
                self.signal_layer[tk][sub_tk] = nn.Tanh().to(self.device)

                self.opt_dict[tk][sub_tk] = torch.optim.Adam(
                    list(self.model_in_dict[tk][sub_tk].parameters())
                    + list(self.model_out_dict[tk][sub_tk].parameters())
                    + list(self.global_transfer_lstm.parameters())
                    + list(self.signal_layer[tk][sub_tk].parameters()),
                    lr=self.opt_lr,
                    amsgrad=self.amsgrad,
                )

                print(
                    tk,
                    sub_tk,
                    self.model_in_dict[tk][sub_tk],
                    self.model_out_dict[tk][sub_tk],
                    self.global_transfer_lstm,
                    self.signal_layer[tk][sub_tk],
                    self.opt_dict[tk][sub_tk],
                )

    def train(self):
        for i in range(self.t_steps):
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

                    in_pred = self.model_in_dict[tk][sub_tk](X_train)

                    (hidden, cell) = self.get_hidden(
                        self.batch_size, self.transfer_layers, self.in_transfer_dim
                    )
                    global_pred, _ = self.global_transfer_lstm(in_pred, (hidden, cell))

                    preds = self.signal_layer[tk][sub_tk](
                        self.model_out_dict[tk][sub_tk](global_pred)
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
                        self.model_in_dict[tk][sub_tk],
                        self.export_path
                        + tk
                        + "_"
                        + sub_tk
                        + "_"
                        + self.export_label
                        + "_intransferlinear.pt",
                    )

                    torch.save(
                        self.model_out_dict[tk][sub_tk],
                        self.export_path
                        + tk
                        + "_"
                        + sub_tk
                        + "_"
                        + self.export_label
                        + "_outtransferlinear.pt",
                    )

                torch.save(
                    self.global_transfer_lstm,
                    self.export_path
                    + tk
                    + "_"
                    + self.export_label
                    + "_globaltransferlstm.pt",
                )

    def predict(self, x_test):
        y_pred = {}

        for tk in self.mtl_list:
            y_pred[tk] = {}

            for sub_tk in self.sub_mtl_list[tk]:
                xflat = x_test[tk][sub_tk].view(1, -1, x_test[tk][sub_tk].size(1))

                with torch.autograd.no_grad():
                    in_pred = self.model_in_dict[tk][sub_tk](xflat[:, :-1])
                    (hidden, cell) = self.get_hidden(
                        1, self.transfer_layers, self.in_transfer_dim
                    )

                    global_pred, _ = self.global_transfer_lstm(in_pred, (hidden, cell))
                    y_pred[tk][sub_tk] = self.signal_layer[tk][sub_tk](
                        self.model_out_dict[tk][sub_tk](global_pred)
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

    def get_hidden(self, batch_size, n_layers, nhi):
        return (
            torch.zeros(n_layers, batch_size, nhi).double().to(self.device),
            torch.zeros(n_layers, batch_size, nhi).double().to(self.device),
        )
