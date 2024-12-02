import copy
import logging
import logging.config
import os
import random
from collections.abc import Iterator
from datetime import datetime
from itertools import islice, product
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import pandera.typing as pa
import pytz
import yaml

from .custom_types import Config, Grid, LSTMInput, NPFloats, Parameters, RecursiveDict

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=np.generic)
T_ = TypeVar("T_", bound=Any)
R = TypeVar("R", bound=np.generic)
TrainTestIds = tuple[list[int], list[int]]


def init_logger(config: RecursiveDict, time_zone: str | None = None) -> None:
    tz = pytz.timezone(time_zone) if time_zone is not None else None
    now = datetime.now(tz=tz)
    log_file = now.strftime(config["handlers"]["file"]["filename"])
    log_path = os.path.dirname(log_file)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    config["handlers"]["file"]["filename"] = log_file
    logging.config.dictConfig(config)
    logger.info("logger instantiated")


def parse_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config: Config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def split_train_test_id(
    id_range: range,
    test_ratio: float,
    random_seed: int | None = None,
) -> TrainTestIds:
    random.seed(random_seed)

    n_tests = int(len(id_range) * test_ratio)

    test_ids = list(random.sample(id_range, n_tests))
    test_id_set = set(test_ids)
    train_ids = [i for i in id_range if i not in test_id_set]

    return train_ids, test_ids


def get_params_combination(grid: Grid) -> Iterator[Parameters]:
    params = list(grid.keys())
    grid_ = copy.deepcopy(grid)
    for hyperparams in product(*grid_.values()):
        param_set = cast(Parameters, {param: hyperparams[i] for i, param in enumerate(params)})

        yield param_set


def rolling_window(series: pa.Series[Any], window: int) -> pa.Series[NPFloats]:
    index = series.index[: -window + 1] if window > 0 else series.index[-window - 1 :]
    window = abs(window)

    return pd.Series(
        (w.values for w in islice(series.rolling(window), window - 1, None)),
        index=index,
    )


def load_google_stock_price(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=["Date"], date_format="%m/%d/%Y").rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    for col, dtype in df.drop(columns=["date"]).dtypes.items():
        if dtype == object:
            df[col] = df[col].str.replace(",", "").astype(float)

    df = df.assign(close=np.where(df.close > df.high, df.close / 2, df.close))

    return df


def to_lstm_input(df: pd.DataFrame) -> LSTMInput:
    return np.array(df.values.tolist())
