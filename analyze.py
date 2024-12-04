#!./env/bin/python
import copy
import math
import os
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lstm.custom_types import NPFloatMatrix
from lstm.utils import load_google_stock_price


def initialize_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def violin_plot(df: pd.DataFrame, file_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 5))

    targets = df_metric.target.unique()
    mses = []
    for target in targets:
        df_sub = df_metric[df_metric.target.eq(target)]
        mses.append(df_sub.validation_mse)

    ax.violinplot(
        mses,
        showmedians=True,
    )

    labels = targets
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    ax.set_yscale("log")

    y_min = 10 ** int(math.log10(df.validation_mse.min()))
    y_max = 10 ** int(math.log10(df.validation_mse.max()) + 1)
    plt.ylim(y_min, y_max)

    plt.xlabel("Target")
    plt.ylabel("MSE")
    plt.savefig(file_path, bbox_inches="tight")


def get_latex_table(df: pd.DataFrame, latex_table_file: str, **to_latex_kwargs: Any) -> None:
    # map the architectures to symbols
    layer_mapping = {
        "[{'layer': 'lstm', 'units': 400}]": r"\architecturea{-0.1cm}",
        "[{'layer': 'lstm', 'units': 200}]": r"\architectureb{-0.1cm}",
        "[{'layer': 'lstm', 'units': 100}]": r"\architecturec{-0.1cm}",
        "[{'layer': 'lstm', 'units': 400, 'return_sequences': True}, {'layer': 'drop_out', 'rate': 0.2}, {'layer': 'lstm', 'units': 200}, {'layer': 'dense_relu', 'units': 64}]": r"\architectured{-0.1cm}",
        "[{'layer': 'lstm', 'units': 200, 'return_sequences': True}, {'layer': 'drop_out', 'rate': 0.2}, {'layer': 'lstm', 'units': 100}, {'layer': 'dense_relu', 'units': 64}]": r"\architecturee{-0.1cm}",
        "[{'layer': 'lstm', 'units': 400, 'return_sequences': True}, {'layer': 'drop_out', 'rate': 0.2}, {'layer': 'lstm', 'units': 200}]": r"\architecturef{-0.1cm}",
        "[{'layer': 'lstm', 'units': 200, 'return_sequences': True}, {'layer': 'drop_out', 'rate': 0.2}, {'layer': 'lstm', 'units': 100}]": r"\architectureg{-0.1cm}",
    }

    mse_formatter = {
        "learning_rate": float,
        "train_mse": round,
        "validation_mse": round,
        "test_mse": round,
    }

    # create raw latex
    latex_table_raw = (
        df.to_latex(
            index=False,
            formatters=mse_formatter,
            **to_latex_kwargs,
        )
        .replace("& Architecture ", "")
        .replace("table", "table*")
    )

    latex_table = []
    row_count = 0
    # modify the style of the table
    for line in latex_table_raw.split("\n"):
        if line.startswith(r"\begin{table"):
            latex_table += [line + "[t]"]
        elif line.startswith(r"\begin{tabul"):
            latex_table += [r"\begin{tabularx}{\textwidth}{lCCcCc}"]
        elif line.startswith("target"):
            latex_table += [r"  \rowcolor{lightgray}"]
            latex_table += [
                r"  \bf Price  & \bf Learning rate & \bf Observed days   & \multicolumn{3}{c}{\bf MSE} \\"
            ]
            latex_table += [r"  \rowcolor{lightgray}"]
            latex_table += [r"   &       &  &                    \bf Train & \bf Validation & \bf Test \\"]
            latex_table += [r"  \bhline"]
        elif line.endswith("rule"):
            continue
        elif line.startswith(("Close", "High", "Low", "Open")):
            if row_count % 2 == 1:
                latex_table += [r"  \evenrow"]

            layer = re.match(r".*(\[.*\])", line).group(1)
            latex_table += ["  " + line.replace(f"{layer} &", "")]

            if row_count % 2 == 1:
                latex_table += [r"  \evenrow"]
            latex_table += [r"  & \multicolumn{5}{l}{" + layer_mapping[layer] + r"} \\"]

            row_count += 1

        elif line.startswith(r"\end{tabu"):
            latex_table[-1] = latex_table[-1][:-3]
            latex_table += [r"\end{tabularx}"]
        else:
            latex_table += [line]

    latex_table_str = "\n".join(
        (f"  {line}" if i > 0 and i < len(latex_table) - 2 else line for i, line in enumerate(latex_table))
    )

    with open(latex_table_file, "w") as f:
        f.write(latex_table_str)


def example_plot(y: dict[str, NPFloatMatrix], predictions: dict[str, NPFloatMatrix], file_path: str) -> None:
    df_plot = []
    n_days = y["test"].shape[1]

    for i in range(n_days):
        df_plot_ = {
            "train": pd.DataFrame(),
            "validation": pd.DataFrame(),
            "test": pd.DataFrame(),
        }

        for set_type, truth in y.items():
            data = {
                "y": truth[1:, i],
                "prediction": predictions[set_type][1:, i],
            }

            df_plot_[set_type] = pd.DataFrame(data)

        df_plot_["train"].index = df_plot_["train"].index + i
        df_plot_["validation"].index = df_plot_["validation"].index + df_plot_["train"].index.max() + 1
        df_plot_["test"].index = df_plot_["test"].index + df_plot_["validation"].index.max() + 1
        df_plot.append(df_plot_)

    _, axes = plt.subplots(nrows=n_days, figsize=(4, 4), sharex=True)

    start_plot = 899
    for i in range(n_days):
        pd.concat((df_plot[i]["train"], df_plot[i]["validation"], df_plot[i]["test"])).iloc[start_plot:].y.plot(
            linewidth=1, ax=axes[i]
        )
        df_plot[i]["train"].iloc[start_plot:].prediction.plot(linewidth=1, ax=axes[i])
        df_plot[i]["validation"].prediction.plot(linewidth=1, ax=axes[i])
        df_plot[i]["test"].prediction.plot(linewidth=1, ax=axes[i])

        axes[i].set_ylabel(f"Day {i + 1}\nprice")

    legend = axes[0].legend(
        ["truth", "predicted train", "predicted validation", "predicted test"],
        loc="upper center",
        bbox_to_anchor=(0.4, 2.05),
        ncol=2,
    )
    plt.xlabel("Day")
    plt.savefig(file_path, bbox_inches="tight")


def simple_plot(df_train: pd.DataFrame, df_test: pd.DataFrame, file_path: str, ylabel: str = "Price") -> None:
    pd.concat((df_train, df_test)).reset_index(drop=True).plot(linewidth=0.5, figsize=(4, 4))

    plt.ylabel(ylabel)
    plt.xlabel("Day")
    plt.savefig(file_path, bbox_inches="tight")


if __name__ == "__main__":
    # create a directory for assets
    asset_dir = "./assets"
    initialize_dir(asset_dir)

    # eda
    simple_plot_file = os.path.join(asset_dir, "simple_plot.pdf")
    df_train = load_google_stock_price("data/Google_Stock_Price_Train.csv")
    df_test = load_google_stock_price("data/Google_Stock_Price_Test.csv")

    prices = ["open", "high", "low", "close"]
    simple_plot(df_train[prices], df_test[prices], simple_plot_file)

    adjust_plot_file = os.path.join(asset_dir, "adjust_plot.pdf")
    df_train = df_train.assign(
        close=np.where(df_train.close > df_train.high, df_train.close / 2, df_train.close)
    )
    simple_plot(df_train[prices], df_test[prices], adjust_plot_file)

    volume_plot_file = os.path.join(asset_dir, "volume_plot.pdf")
    simple_plot(df_train[["volume"]], df_test[["volume"]], volume_plot_file, "Volume")

    # overall mse plot
    df_metric = pd.read_csv("./metrics.csv", sep="|")

    violin_plot_file = os.path.join(asset_dir, "violin_plot.pdf")
    violin_plot(df_metric, violin_plot_file)

    # write to latex table
    use_cols = [
        "target",
        "learning_rate",
        "look_back_day",
        "architecture",
        "train_mse",
        "validation_mse",
        "test_mse",
    ]

    latex_table_file = "./latex/best-performance.tex"
    df_best = (
        df_metric.groupby(["target"])
        .validation_mse.min()
        .reset_index()
        .merge(df_metric, on=["target", "validation_mse"])[use_cols]
        .assign(target=lambda x: x.target.str.title())
    )
    get_latex_table(
        df_best,
        latex_table_file,
        caption="The best performance of each price on the validation set",
        label="tab:price-performance",
    )

    # example plot
    example_file = os.path.join(asset_dir, "example_plot.pdf")
    y: dict[str, NPFloatMatrix] = {
        "train": np.array(()),
        "validation": np.array(()),
        "test": np.array(()),
    }
    predictions = copy.deepcopy(y)

    for set_type in y.keys():
        predictions[set_type] = np.load(f"./results/target=high/prediction_{set_type}.npy")
        y[set_type] = np.load(f"./results/target=high/truth_{set_type}.npy")

    example_plot(y, predictions, example_file)
