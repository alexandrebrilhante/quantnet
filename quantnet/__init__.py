import random
import time

import numpy as np
import pandas as pd

from quantnet.baselines import (
    BuyAndHold,
    CrossSectionalMomentum,
    RiskParity,
    TimeSeriesMomentum,
)
from quantnet.utils import calc_transaction_costs


class QuantNet:
    def __init__(self, seed=999999999):
        np.random.seed(seed)
        random.seed(seed)

        self.data_config = {
            "data_path": ".\\Tasks\\",
            "region": ["Asia and Pacific", "Europe", "Americas", "MEA"],
            "Europe": [
                "Europe_AEX",
                "Europe_ASE",
                "Europe_ATX",
                "Europe_BEL20",
                "Europe_BUX",
                "Europe_BVLX",
                "Europe_CAC",
                "Europe_CYSMMAPA",
                "Europe_DAX",
                "Europe_HEX",
                "Europe_IBEX",
                "Europe_ISEQ",
                "Europe_KFX",
                "Europe_OBX",
                "Europe_OMX",
                "Europe_SMI",
                "Europe_UKX",
                "Europe_VILSE",
                "Europe_WIG20",
                "Europe_XU100",
                "Europe_SOFIX",
                "Europe_SBITOP",
                "Europe_PX",
                "Europe_CRO",
            ],
            "Asia and Pacific": [
                "Asia and Pacific_AS51",
                "Asia and Pacific_FBMKLCI",
                "Asia and Pacific_HSI",
                "Asia and Pacific_JCI",
                "Asia and Pacific_KOSPI",
                "Asia and Pacific_KSE100",
                "Asia and Pacific_NIFTY",
                "Asia and Pacific_NKY",
                "Asia and Pacific_NZSE50FG",
                "Asia and Pacific_PCOMP",
                "Asia and Pacific_STI",
                "Asia and Pacific_SHSZ300",
                "Asia and Pacific_TWSE",
            ],
            "Americas": [
                "Americas_IBOV",
                "Americas_MEXBOL",
                "Americas_MERVAL",
                "Americas_SPTSX",
                "Americas_SPX",
                "Americas_RTY",
            ],
            "MEA": [
                "MEA_DFMGI",
                "MEA_DSM",
                "MEA_EGX30",
                "MEA_FTN098",
                "MEA_JOSMGNFF",
                "MEA_KNSMIDX",
                "MEA_KWSEPM",
                "MEA_MOSENEW",
                "MEA_MSM30",
                "MEA_NGSE30",
                "MEA_PASISI",
                "MEA_SASEIDX",
                "MEA_SEMDEX",
                "MEA_TA-35",
                "MEA_TOP40",
            ],
            "additional_data_path": "_all_assets_data.pkl.gz",
        }

        self.problem_config = {
            "export_path": "./Results/",
            "val_period": 0,
            "holdout_period": 756,
        }

        self.model_config = {
            "baseline": "risk_parity",
            "buy_and_hold": {},
            "risk_parity": {"window": 252},
            "ts_mom": {"window": 252},
            "csec_mom": {"window": 252, "fraction": 0.33},
        }

        self.export_label = (
            "val_period_"
            + str(self.problem_config["val_period"])
            + "_testperiod_"
            + str(self.problem_config["holdout_period"])
            + "_baseline_"
            + self.model_config["baseline"]
        )

        self.data_config["export_label"] = self.export_label
        self.problem_config["export_label"] = self.export_label
        self.model_config["export_label"] = self.export_label
        self.model_config["export_path"] = self.problem_config["export_path"]

        self.X_train_tasks, self.X_val_tasks, self.X_test_tasks = self.get_data(
            self.data_config, self.problem_config, self.model_config
        )

    def train(self):
        if self.model_config["baseline"] == "buy_and_hold":
            trad_strat = BuyAndHold(self.X_train_tasks, self.model_config)
            add_label = [""] * len(self.data_config["region"])

        elif self.model_config["baseline"] == "risk_parity":
            trad_strat = RiskParity(self.X_train_tasks, self.model_config)
            add_label = [""] * len(self.data_config["region"])

        elif self.model_config["baseline"] == "ts_mom":
            trad_strat = TimeSeriesMomentum(self.X_train_tasks, self.model_config)
            add_label = [""] * len(self.data_config["region"])

        elif self.model_config["baseline"] == "csec_mom":
            trad_strat = CrossSectionalMomentum(self.X_train_tasks, self.model_config)
            add_label = [""] * len(self.data_config["region"])

        to_add_label = {}

        for lab, region in zip(add_label, self.data_config["region"]):
            to_add_label[region] = lab

        start = time()
        trad_strat.train()

        print(time() - start)

        self.X_train_signal = trad_strat.predict(self.X_train_tasks)
        self.X_val_signal = trad_strat.predict(self.X_val_tasks)
        self.X_test_signal = trad_strat.predict(self.X_test_tasks)

    def predict(self):
        results = True

        for region in self.data_config["region"]:
            region_task_paths = [
                t + "_all_assets_data.pkl.gz" for t in self.data_config[region]
            ]

            metrics = True

            for tk, tk_path in zip(self.data_config[region], region_task_paths):
                pred_train = self.X_train_signal[region][tk][:-1, :]
                pred_val = self.X_val_signal[region][tk][:-1, :]
                pred_test = self.X_test_signal[region][tk][:-1, :]

                Y_train = self.X_train_tasks[region][tk][1:, :]
                Y_val = self.X_val_tasks[region][tk][1:, :]
                Y_test = self.X_test_tasks[region][tk][1:, :]

                df_train_ret = np.multiply(
                    pred_train, Y_train
                ) - calc_transaction_costs(pred_train)
                df_val_ret = np.multiply(pred_val, Y_val) - calc_transaction_costs(
                    pred_val
                )
                df_test_ret = np.multiply(pred_test, Y_test) - calc_transaction_costs(
                    pred_test
                )

                df = pd.read_pickle(self.data_config["data_path"] + tk_path)
                df_train_ret = pd.DataFrame(df_train_ret, columns=df.columns)
                df_train_metrics = self.compute_performance_metrics(df_train_ret)
                df_train_metrics["exchange"] = tk

                df_val_ret = pd.DataFrame(df_val_ret, columns=df.columns)
                df_val_metrics = self.compute_performance_metrics(df_val_ret)
                df_val_metrics["exchange"] = tk

                df_test_ret = pd.DataFrame(df_test_ret, columns=df.columns)
                df_test_metrics = self.compute_performance_metrics(df_test_ret)
                df_test_metrics["exchange"] = tk

                if metrics:
                    all_df_train_metrics = df_train_metrics.copy()
                    all_df_val_metrics = df_val_metrics.copy()
                    all_df_test_metrics = df_test_metrics.copy()
                    z = False
                else:
                    all_df_train_metrics = pd.concat(
                        [all_df_train_metrics, df_train_metrics], axis=0
                    )
                    all_df_val_metrics = pd.concat(
                        [all_df_val_metrics, df_val_metrics], axis=0
                    )
                    all_df_test_metrics = pd.concat(
                        [all_df_test_metrics, df_test_metrics], axis=0
                    )

        all_df_train_metrics["region"] = region
        all_df_train_metrics["set"] = "train"
        all_df_val_metrics["region"] = region
        all_df_val_metrics["set"] = "val"
        all_df_test_metrics["region"] = region
        all_df_test_metrics["set"] = "test"

        pd.concat(
            [all_df_train_metrics, all_df_val_metrics, all_df_test_metrics], axis=0
        ).to_csv(
            self.problem_config["export_path"]
            + region
            + "_"
            + self.problem_config["export_label"]
            + self.to_add_label[region]
            + ".csv"
        )

        if results:
            global_df_test_metrics = all_df_test_metrics.copy()
            results = False
        else:
            global_df_test_metrics = pd.concat(
                [global_df_test_metrics, all_df_test_metrics.copy()], axis=0
            )
