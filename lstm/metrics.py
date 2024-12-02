import os
from datetime import datetime
from typing import Iterable, Mapping

from sklearn.metrics import mean_squared_error

from .custom_types import SimpleMetrics, Y


class Metrics:
    def __init__(self, metric_file: str, set_types: Iterable[str]) -> None:
        self.metric_file = metric_file
        self.set_types = set_types
        self.parameters = ["learning_rates", "look_back_days", "predict_days", "architectures"]

        self.write_header()

    def write_header(self) -> None:
        if not os.path.isfile(self.metric_file):
            metrics = ["mse"]

            metric_cols = "|".join(
                [f"{set_type}_{metric}" for set_type in self.set_types for metric in metrics]
            )
            param_cols = "|".join(map(lambda x: x[:-1], self.parameters))

            with open(self.metric_file, "a") as f:
                f.write(f"saved_at|{metric_cols}|{param_cols}|target\n")

    def write(self, parameters: Mapping[str, object], simple_metrics: SimpleMetrics, target: str) -> None:
        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metric_file, "a") as f:
            serialized_params = "|".join(map(str, (parameters[p] for p in self.parameters)))
            serialized_metrics = "|".join(map(str, (simple_metrics[s] for s in self.set_types)))

            f.write(f"{saved_at}|{serialized_metrics}|{serialized_params}|{target}\n")

    @staticmethod
    def write_target(y: Y, model_name: str, target_file: str) -> None:
        with open(target_file, "a") as f:
            for set_type, target in y.items():
                serialized_y = "|".join(map(str, target))
                f.write(f"{model_name}|{set_type}|{serialized_y}\n")

    @staticmethod
    def get_mse(truth: Y, predictions: Y) -> SimpleMetrics:
        return {set_type: mean_squared_error(t, predictions[set_type]) for set_type, t in truth.items()}
