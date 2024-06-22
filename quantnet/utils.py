import numpy as np
import pandas as pd
import torch
from empyrical import (
    annual_return,
    annual_volatility,
    calmar_ratio,
    downside_risk,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)
from scipy.stats import kurtosis, skew


def compute_performance_metrics(df):
    pf_metrics = [
        sharpe_ratio,
        calmar_ratio,
        max_drawdown,
        annual_return,
        annual_volatility,
        sortino_ratio,
        downside_risk,
        value_at_risk,
        tail_ratio,
        skew,
        kurtosis,
    ]

    pf_metrics_labels = [
        "SR",
        "CR",
        "MDD",
        "AnnRet",
        "AnnVol",
        "SortR",
        "DownRisk",
        "VaR",
        "TailR",
        "Skew",
        "Kurt",
    ]

    df_metrics = pd.DataFrame(index=range(df.shape[1]), columns=pf_metrics_labels)

    for pf, pf_label in zip(pf_metrics, pf_metrics_labels):
        df_metrics[pf_label] = np.array(pf(df))

    df_metrics.index = df.columns

    return df_metrics


def get_data(data_config, problem_config, model_config):
    X_train_tasks, X_val_tasks, X_test_tasks = {}, {}, {}

    for region in data_config["region"]:
        region_task_paths = [t + "_all_assets_data.pkl.gz" for t in data_config[region]]
        X_train_tasks[region], X_val_tasks[region], X_test_tasks[region] = {}, {}, {}

        for tk_path, tk in zip(region_task_paths, data_config[region]):
            df = pd.read_pickle(data_config["data_path"] + tk_path)
            df_train = df.iloc[
                : -(problem_config["val_period"] + problem_config["holdout_period"])
            ]

            if problem_config["val_period"] != 0:
                df_val = df.iloc[
                    -(
                        problem_config["val_period"] + problem_config["holdout_period"]
                    ) : -problem_config["holdout_period"]
                ]
            else:
                df_val = df.iloc[
                    : -(problem_config["val_period"] + problem_config["holdout_period"])
                ]

            df_test = df.iloc[-problem_config["holdout_period"] :]

            X_train_tasks[region][tk] = torch.from_numpy(df_train.values).to(
                model_config["device"]
            )

            X_val_tasks[region][tk] = torch.from_numpy(df_val.values).to(
                model_config["device"]
            )

            X_test_tasks[region][tk] = torch.from_numpy(df_test.values).to(
                model_config["device"]
            )

    return X_train_tasks, X_val_tasks, X_test_tasks


def calc_transaction_costs(signal):
    slip = 0.0005 * 0.00
    bp = 0.0020 * 0.00
    tc = np.abs(signal[1:, :] - signal[:-1, :]) * (bp + slip)
    tc = np.concatenate([np.zeros(signal.shape[1]).reshape(1, -1), tc], axis=0)

    return tc
