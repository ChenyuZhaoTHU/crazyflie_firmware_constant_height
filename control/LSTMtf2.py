import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from scipy.signal import stft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import scipy.signal as signal_p





def load_data():
    # on00all = np.zeros((30,200,9))
    train_data = np.zeros((60,400,9))
    
    for i in range(30):
        train_data[i,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data2/on00_'+str(i+1)+'.txt')
    # on45all = np.zeros((30,200,9))
    for i in range(30):
        train_data[i+30,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data2/on45_'+str(i+1)+'.txt')
    # for i in range(30):
    #     print(i)
    #     train_data[i+60,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on00_'+str(i+1)+'.txt')

    # FILTER
    # Define filter parameters
    fs = 100  # Sampling frequency
    fc = 30  # Cutoff frequency
    order = 6  # Filter order
    # Create low-pass Butterworth filter
    b, a = signal_p.butter(order, fc / (fs / 2), btype='low')

    # Apply filter to signal data
    signalf = signal_p.filtfilt(b, a, train_data[:,:,:7], axis=1)

    train_data = signalf
    # train_data = train_data[:,80:105,:]

    # print(np.shape(train_data))
    # for p in range(np.shape(train_data)[0]):
    #     train_data[p,:,4] = train_data[p,:,4] - train_data[p,:,3]
    #     train_data[p,:,5] = train_data[p,:,5] - train_data[p,:,3]
    #     train_data[p,:,6] = train_data[p,:,6] - train_data[p,:,3]

    # data = np.random.random((90, 200, 7))

    # 计算第三维度的最大值和最小值
    min_vals = np.min(train_data, axis=0, keepdims=True)
    max_vals = np.max(train_data, axis=0, keepdims=True)

    # 对第三维度进行归一化
    train_data = (train_data - min_vals) / (max_vals - min_vals)


    # train_data = np.delete(train_data, [3], 2)
    train_stft = np.zeros((60,11,101,np.shape(train_data)[2]))
    for i in range(np.shape(train_data)[0]):
        for j in range(np.shape(train_data)[2]):
            # print(np.shape(train_data[i,:,j]))
            t,f,Zxx = stft(train_data[i,:,j], fs=200, nperseg=20, noverlap=16)
            # print(np.shape(Zxx))
            train_stft[i,:,:,j] = Zxx
    # train_data = train_data[:,:100,:]
    # train_data = tf.expand_dims(train_data,-1)
    # train_data = np.delete(train_data, [7,8], 2)


    # print(np.shape(train_data[7,:,0,0]))
    # print(train_data[7,:,0,0])
    # plt.figure()
    # plt.plot(train_data[7,:,0,0])
    # plt.show()



    label1 = np.zeros((60,2))
    label1[:30,0] = 1
    label1[30:60,1] = 1
    # label1[60:,2] = 1
    print(label1)
    # return label1

    perm = np.random.permutation(train_data.shape[0])
    X_shuffled = train_data[perm,:,:]
    y_shuffled = label1[perm,:]


    # X_shuffled = tf.expand_dims(X_shuffled,-1)



    return X_shuffled, y_shuffled




def load_data2():
    # on00all = np.zeros((30,200,9))
    train_data = np.zeros((90,200,9))
    
    for i in range(30):
        train_data[i,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on00_'+str(i+1)+'.txt')
    # on45all = np.zeros((30,200,9))
    for i in range(30):
        train_data[i+30,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on45_'+str(i+1)+'.txt')
    for i in range(30):
        print(i)
        train_data[i+60,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on135_'+str(i+1)+'.txt')

    train_data = train_data[:,95:105,:]

    # print(np.shape(train_data))
    for p in range(np.shape(train_data)[0]):
        train_data[p,:,4] = train_data[p,:,4] - train_data[p,:,3]
        train_data[p,:,5] = train_data[p,:,5] - train_data[p,:,3]
        train_data[p,:,6] = train_data[p,:,6] - train_data[p,:,3]

    # data = np.random.random((90, 200, 7))

    # 计算第三维度的最大值和最小值
    # min_vals = np.min(train_data, axis=0, keepdims=True)
    # max_vals = np.max(train_data, axis=0, keepdims=True)

    # # 对第三维度进行归一化
    # train_data = (train_data - min_vals) / (max_vals - min_vals)


    train_data = np.delete(train_data, [0,1,2,3,7,8], 2)
    train_stft = np.zeros((90,6,3,np.shape(train_data)[2]))
    for i in range(np.shape(train_data)[0]):
        for j in range(np.shape(train_data)[2]):
            # print(np.shape(train_data[i,:,j]))
            t,f,Zxx = stft(train_data[i,:,j], fs=200, nperseg=10, noverlap=5)
            print(np.shape(Zxx))
            train_stft[i,:,:,j] = Zxx
    # train_data = train_data[:,:100,:]
    # train_data = tf.expand_dims(train_data,-1)
    # train_data = np.delete(train_data, [7,8], 2)


    # print(np.shape(train_data[7,:,0,0]))
    # print(train_data[7,:,0,0])
    # plt.figure()
    # plt.plot(train_data[7,:,0,0])
    # plt.show()
    return train_data

def gen_label():
    label1 = np.zeros((90,3))
    label1[:30,0] = 1
    label1[30:60,1] = 1
    label1[60:,2] = 1
    print(label1)
    return label1

class CNN(object):
    def __init__(self, train_x):
    # def create_model(train_x):
        model = Sequential()
        # model.add(tf.keras.layers.Conv2D(input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]),
        #                             filters=2, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=1))
        # model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=1))
        # model.add(tf.keras.layers.Flatten())
        # train_x = train_x.reshape(train_x.shape[0],200,)
        model.add(LSTM(20, input_shape=(train_x.shape[1], train_x.shape[2]), activation='relu', return_sequences=True))
        model.add(LSTM(20, activation='relu'))
        
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        # model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))

        

        model.add(tf.keras.layers.Dense(32))
        model.add(tf.keras.layers.Dense(16))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
        model.compile(loss='mse', optimizer='adam', metrics='accuracy')
        model.summary()
        self.model = model



    # def __init__(self, train_x):
    #     model = Sequential()
    #     model.add(tf.keras.layers.LSTM(units=10, input_shape=(train_x.shape[1], train_x.shape[2])))
    #     model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
    #     model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
    #     model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
    #     model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    #     model.summary()
    #     self.model = model


# def create_dataset():
    





class Train():
    def __init__(self):
        # self.input1_train, self.input2_train, self.label1, self.label2 = create_dataset()
        self.input1_train, self.label1 = load_data()
        self.model_usr1 = CNN(self.input1_train)
        
    def train(self):
        history_usr1 = self.model_usr1.model.fit(self.input1_train[3:57,:,:], self.label1[3:57,:], epochs=200, batch_size = 8, 
                            validation_split=0.30, verbose=1, shuffle=True)
      
        os.makedirs('savedmodel', exist_ok=True)
        self.model_usr1.model.save('/home/nuci7/project/cf2/crazyflie-firmware/control/savedmodel/lstm_usr1_model.h5')
        print(history_usr1.history.keys())
        print(history_usr1.history['loss'],history_usr1.history['val_loss'])
        plt.figure(1)
        plt.plot(history_usr1.history['loss'])
        plt.plot(history_usr1.history['val_loss'])
 
        plt.grid(True)
        plt.show()
        return self.input1_train, self.label1


def test_n(test_data, test_label):
    model_pre1 = tf.keras.models.load_model('/home/nuci7/project/cf2/crazyflie-firmware/control/savedmodel/lstm_usr1_model.h5')
    # print(np.shape(load_data()[2:3,:,:,:]))

    # test_data, test_label = load_data()
    ooout1 = model_pre1.predict(test_data[0:3,:,:])
    ooout2 = model_pre1.predict(test_data[57:,:,:])
    # ooout3 = model_pre1.predict(load_data()[85:90,:,:,:])
    # print(np.shape(test_label1))
    # print(np.shape(ooout1))
    print(ooout1,'\n  group1',test_label[0:3,:])
    print(ooout2,'\n  group2',test_label[57:,:])
    # print(ooout3)





if __name__ == "__main__":
    # littletest()
    app = Train()
    test_data, test_label = app.train()
    
    # load_data()


    test_n(test_data, test_label)
    # load_data()
    # gen_label()
    # print(gen_label())
    # test()

