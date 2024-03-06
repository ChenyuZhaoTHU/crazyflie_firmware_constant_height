import math
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, BatchNormalization, Dense
from scipy.signal import stft

def load_data(file_paths):
    train_data = np.zeros((90, 200, 9))

    for i, file_path in enumerate(file_paths):
        train_data[i] = np.loadtxt(file_path)

    min_vals = np.min(train_data, axis=0, keepdims=True)
    max_vals = np.max(train_data, axis=0, keepdims=True)

    train_data = (train_data - min_vals) / (max_vals - min_vals)
    train_data = np.delete(train_data, [3, 7, 8], 2)

    train_stft = np.zeros((90, 51, 11, np.shape(train_data)[2]))
    for i in range(np.shape(train_data)[0]):
        for j in range(np.shape(train_data)[2]):
            _, _, Zxx = stft(train_data[i, :, j], fs=100, nperseg=100, noverlap=80)
            train_stft[i, :, :, j] = Zxx

    return train_stft
def gen_label():
    label1 = np.zeros((90, 3))
    label1[:30, 0] = 1
    label1[30:60, 1] = 1
    label1[60:, 2] = 1

    return label1

class CNN(object):
    def __init__(self, train_x):
        self.model = Sequential([
            Conv2D(input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]), filters=2, kernel_size=(2, 2),
                   strides=(1, 1), padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2], strides=1),
            Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2], strides=1),
            Flatten(),
            BatchNormalization(center=True, scale=True),
            Dense(200, activation='relu'),
            BatchNormalization(center=True, scale=True),
            Dense(20, activation='relu'),
            Dense(3, activation='softmax')
        ])
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


class Train:
    def __init__(self):
        self.input1_train = load_data()
        self.label1 = gen_label()
        self.model_usr1 = CNN(self.input1_train)

    def train(self):
        history_usr1 = self.model_usr1.model.fit(
            self.input1_train[3:85, :, :, :],
            self.label1[3:85, :],
            epochs=900,
            batch_size=2,
            validation_split=0.35,
            verbose=1,
            shuffle=True
        )

        os.makedirs('savedmodel', exist_ok=True)
        self.model_usr1.model.save('savedmodel/cnn_usr1_model.h5')

        plt.plot(history_usr1.history['loss'])
        plt.plot(history_usr1.history['val_loss'])
        plt.grid(True)
        plt.show()


def test_n():
    model_pre1 = tf.keras.models.load_model('savedmodel/cnn_usr1_model.h5')
    test_data = load_data(test_file_paths)  # Provide test file paths
    predictions = model_pre1.predict(test_data)
    print(predictions)


if __name__ == "__main__":
    # littletest()
    app = Train()
    app.train()
    
    # load_data()


    test_n()