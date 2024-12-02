from typing import Any

import tensorflow as tf

from .custom_types import Layers, NPFloatMatrix, copy_args

tf_initializers = tf.keras.initializers
tf_layers = tf.keras.layers


class LSTM:
    def __init__(
        self,
        layers: Layers,
        learning_rate: float,
        input_shape: tuple[int, ...],
        output_shape: int,
        loss: str = "mean_squared_error",
        random_seed: int | None = None,
    ) -> None:
        # set seed for possible reproducibility
        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)

        self.is_model_compiled = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = loss

        self._model = self.init_model(layers, input_shape, output_shape)

    def init_model(self, layers: Layers, input_shape: tuple[int, ...], output_shape: int) -> tf.keras.Model:
        # initialise model with additional layers
        input_ = tf.keras.layers.Input(shape=input_shape)

        x = TFLayer.get_tensor_from_config(layers, input_)
        y = TFLayer.dense(output_shape)(x)

        return tf.keras.Model(inputs=input_, outputs=y)

    @property
    def model(self) -> tf.keras.Model:
        if not self.is_model_compiled:
            self._model.compile(optimizer=self.optimizer, loss=self.loss)
            self.is_model_compiled = True

        return self._model

    @copy_args(tf.keras.Model.fit)
    def fit(self, *args, **kwargs: Any) -> None:
        # fit with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        kwargs["callbacks"] = [early_stopping]

        self.model.fit(*args, **kwargs)

    @copy_args(tf.keras.Model.predict)
    def predict(self, *args, **kwargs: Any) -> NPFloatMatrix:
        return self.model.predict(*args, **kwargs)


class TFLayer:
    @staticmethod
    def dense(units: int, **kwargs: Any) -> tf_layers.Dense:
        return tf_layers.Dense(units, kernel_initializer=tf_initializers.HeNormal(), **kwargs)

    @staticmethod
    def dense_relu(units: int, **kwargs: Any) -> tf_layers.Dense:
        return TFLayer.dense(units, activation="relu", **kwargs)

    @staticmethod
    def mlp(x: tf.Tensor, units: int, n: int, **_: Any) -> tf.Tensor:
        if n == 0:
            return x

        layer = TFLayer.dense_relu(units)(x)
        return TFLayer.mlp(layer, units, n - 1)

    @staticmethod
    def get_tensor_from_config(layers: Layers, input_: tf.Tensor) -> tf.Tensor:
        # create a tensor from a config
        tensor = input_
        for layer in layers:
            kwargs = layer.copy()
            layer_name = kwargs.pop("layer", None)

            match layer_name:
                case "dense":
                    tensor = TFLayer.dense(**kwargs)(tensor)
                case "dense_relu":
                    tensor = TFLayer.dense_relu(**kwargs)(tensor)
                case "mlp":
                    tensor = TFLayer.mlp(tensor, **kwargs)
                case "lstm":
                    tensor = tf_layers.LSTM(**kwargs)(tensor)
                case "drop_out":
                    tensor = tf_layers.Dropout(**kwargs)(tensor)
                case _:
                    raise NotImplementedError(f"Unknown layer {layer_name}")

        return tensor
