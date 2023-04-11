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

        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE, name="targets")
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy")

    def train(self, X, y):
        print(f"X {X.shape}")
        print(f"y {y.shape}")
        self.model.fit({"input": X}, {"targets": y}, epochs=5, verbose=2, steps_per_epoch=500)

    def predict(self, prev_observations):
        return np.argmax(self.model.predict(prev_observations.reshape(-1, len(prev_observations), 1))[0])