import tensorflow as tf
import numpy as np

LEARNING_RATE = 2.33e-4

class Agent():
    def __init__(self, input_size) -> None:
        self.model = tf.keras.Sequential()
        # Input layer
        self.model.add(tf.keras.layers.Input(shape=[None, input_size, 1], name="input"))

        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.8))

        self.model.add(tf.keras.layers.Dense(256, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.8))

        self.model.add(tf.keras.layers.Dense(512, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.8))

        self.model.add(tf.keras.layers.Dense(256, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.8))

        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.8))

        # Output layer
        self.model.add(tf.keras.layers.Dense(2, activation="softmax"))

        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE, loss="categorical_crossentropy", name="targets")
        self.model.compile(optimizer=optimizer)

    def train(self, X, y):
        self.model.fit({"input": X}, {"targets": y}, n_epochs=5, snapshot_step=500, show_metric=True, run_id="pendelum")

    def predict(self, prev_observations):
        return np.argmax(self.model.predict(prev_observations.reshape(-1, len(prev_observations), 1))[0])