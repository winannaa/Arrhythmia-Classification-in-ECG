import tensorflow as tf
from tensorflow.keras import layers, models


class CustomGRU(layers.Layer):
    def __init__(self, units, return_sequences=False, dropout=0.0):
        super(CustomGRU, self).__init__()

        self.gru = layers.GRU(
            units,
            return_sequences=return_sequences,
            dropout=dropout
        )

    def call(self, inputs):
        return self.gru(inputs)


def build_gru_model(input_shape, units=64, dropout=0.2, num_classes=5):

    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))

    model.add(layers.GRU(units, return_sequences=False))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def build_bigru_model(input_shape, units=64, dropout=0.2, num_classes=5):

    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))

    model.add(layers.Bidirectional(
        layers.GRU(units, return_sequences=False)
    ))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
