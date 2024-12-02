#!./env/bin/python
import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lstm.custom_types import Y
from lstm.lstm import LSTM
from lstm.metrics import Metrics
from lstm.utils import (
    get_params_combination,
    init_logger,
    load_google_stock_price,
    parse_config,
    rolling_window,
    to_lstm_input,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = parse_config("./config.yaml")
    init_logger(config["logging"], config["time_zone"])

    # reading data
    df_raw = load_google_stock_price("data/Google_Stock_Price_Train.csv")
    df_test_raw = load_google_stock_price("data/Google_Stock_Price_Test.csv")
    features = ["open", "high", "low", "close", "volume"]
    targets = features[:-1]
    validation_ratio = 0.15

    best_mse = {target: np.inf for target in targets}
    best_truth: dict[str, Y] = {target: {} for target in targets}
    best_prediction = copy.deepcopy(best_truth)

    metrics = Metrics("./metrics.csv", ["train", "validation", "test"])
    for params in get_params_combination(config["grid"]):
        logger.info(f"training: {params}")

        # preparing data for lstm input
        df = pd.DataFrame(index=df_raw.index)
        df_test_concat = pd.concat((df_raw[-params["look_back_days"] :], df_test_raw)).reset_index(drop=True)
        df_test = pd.DataFrame(index=df_test_concat.index)
        for feature in features:
            df[feature] = rolling_window(df_raw[feature], params["look_back_days"])
            df_test[feature] = rolling_window(df_test_concat[feature], params["look_back_days"])

        # simulate target to drop na
        df = df.assign(target=rolling_window(df_raw.close.shift(1).dropna(), -params["predict_days"])).dropna()
        df_test = df_test.assign(
            target=rolling_window(df_test_concat.close.shift(1).dropna(), -params["predict_days"])
        ).dropna()
        dates = df_raw.iloc[df.index].date
        dates_test = df_test_concat.iloc[df_test.index].date
        validation_size = int(df.shape[0] * validation_ratio)
        train_size = df.shape[0] - validation_size - params["predict_days"] + 1

        # scale features
        scaler = StandardScaler(copy=False)
        lstm_train = scaler.fit_transform(
            to_lstm_input(df[features].iloc[:train_size]).reshape(train_size, -1)
        ).reshape(train_size, len(features), params["look_back_days"])

        lstm_validation = scaler.transform(
            to_lstm_input(df[features].iloc[-validation_size:]).reshape(validation_size, -1)
        ).reshape(validation_size, len(features), params["look_back_days"])

        lstm_test = scaler.transform(to_lstm_input(df_test[features]).reshape(df_test.shape[0], -1)).reshape(
            df_test.shape[0], len(features), params["look_back_days"]
        )

        for target_col in targets:
            logger.info(f" - {target_col}")
            target = rolling_window(df_raw[target_col].shift(1).dropna(), -params["predict_days"])[df.index]

            y = {
                "train": np.array(target.iloc[:train_size].values.tolist()),
                "validation": np.array(target.iloc[-validation_size:].values.tolist()),
                "test": np.array(
                    rolling_window(df_test_concat[target_col].shift(1).dropna(), -params["predict_days"])[
                        df_test.index
                    ].values.tolist()
                ),
            }

            lstm = LSTM(
                params["architectures"],
                params["learning_rates"],
                lstm_train.shape[1:],
                params["predict_days"],
                random_seed=config["random_seed"],
            )

            lstm.fit(
                lstm_train,
                y["train"],
                epochs=10000000,
                batch_size=train_size,
                validation_data=(lstm_validation, y["validation"]),
                verbose=0,
            )

            predictions = {
                "train": lstm.predict(lstm_train),
                "validation": lstm.predict(lstm_validation),
                "test": lstm.predict(lstm_test),
            }

            mses = Metrics.get_mse(y, predictions)
            metrics.write(params, mses, target_col)

            # get outputs from best models
            if mses["validation"] < best_mse[target_col]:
                best_mse[target_col] = mses["validation"]
                best_prediction[target_col] = predictions
                best_truth[target_col] = y

    # write outputs from best models to files
    base_output_path = "./results"
    for target_col in targets:
        target_path = os.path.join(base_output_path, f"target={target_col}")
        Path(target_path).mkdir(parents=True, exist_ok=True)

        for set_type in best_prediction[target_col].keys():
            truth_file = os.path.join(target_path, f"truth_{set_type}.npy")
            np.save(truth_file, best_truth[target_col][set_type], allow_pickle=False)

            prediction_file = os.path.join(target_path, f"prediction_{set_type}.npy")
            np.save(prediction_file, best_prediction[target_col][set_type], allow_pickle=False)
