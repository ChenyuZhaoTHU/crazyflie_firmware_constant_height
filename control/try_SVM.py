from sklearn.svm import SVC
import numpy as np
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
        train_data[i+60,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on30_'+str(i+1)+'.txt')

    train_data = train_data[:,95:105,:]

    # print(np.shape(train_data))
    for p in range(np.shape(train_data)[0]):
        train_data[p,:,4] = train_data[p,:,4] - train_data[p,:,3]
        train_data[p,:,5] = train_data[p,:,5] - train_data[p,:,3]
        train_data[p,:,6] = train_data[p,:,6] - train_data[p,:,3]

    # data = np.random.random((90, 200, 7))

    # 计算第三维度的最大值和最小值
    min_vals = np.min(train_data, axis=1, keepdims=True)
    max_vals = np.max(train_data, axis=1, keepdims=True)

    # 对第三维度进行归一化
    train_data = (train_data - min_vals) / (max_vals - min_vals)


    train_data = np.delete(train_data, [3,7,8], 2)
    train_stft = np.zeros((90,6,3,np.shape(train_data)[2]))
    for i in range(np.shape(train_data)[0]):
        for j in range(np.shape(train_data)[2]):
            # print(np.shape(train_data[i,:,j]))
            t,f,Zxx = stft(train_data[i,:,j], fs=200, nperseg=10, noverlap=2)
            print(np.shape(Zxx))
            train_stft[i,:,:,j] = Zxx
    # train_data = train_data[:,:100,:]
    train_data = tf.expand_dims(train_data,-1)
    # train_data = np.delete(train_data, [7,8], 2)


    # print(np.shape(train_data[7,:,0,0]))
    # print(train_data[7,:,0,0])
    # plt.figure()
    # plt.plot(train_data[7,:,0,0])
    # plt.show()
    return train_stft.reshape(90,-1)



def load_data():
    # on00all = np.zeros((30,200,9))
    train_data = np.zeros((90,200,9))
    for i in range(30):
        train_data[i,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on00_'+str(i+1)+'.txt')
    # on45all = np.zeros((30,200,9))
    for i in range(30):
        train_data[i+30,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on45_'+str(i+1)+'.txt')
    for i in range(30):
        train_data[i+60,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on135_'+str(i+1)+'.txt')





    # print(np.shape(train_data))
    for p in range(np.shape(train_data)[0]):
        train_data[p,:,4] = train_data[p,:,4] - train_data[p,:,3]
        train_data[p,:,5] = train_data[p,:,5] - train_data[p,:,3]
        train_data[p,:,6] = train_data[p,:,6] - train_data[p,:,3]
    train_data = np.delete(train_data, [3,4,5,6,7,8], 2)
    # train_data = train_data[:,:100,:]
    # train_data = tf.expand_dims(train_data,-1)
    # train_data = np.delete(train_data, [7,8], 2)


    # print(np.shape(train_data[7,:,0,0]))
    # print(train_data[7,:,0,0])
    # plt.figure()
    # plt.plot(train_data[7,:,0,0])
    # plt.show()
    return train_data.reshape(90,-1)

def gen_label():
    label1 = np.zeros((90,3))
    label1[:30,0] = 1
    label1[30:60,1] = 1
    label1[60:,2] = 1
    print(label1)
    return label1





train_data = load_data2()
label1 = np.array([0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
# 准备数据

# 创建SVM分类器并进行训练
svm = SVC(kernel='linear', decision_function_shape='ovr')
svm.fit(train_data[7:87], label1[7:87])

# 进行预测
# X_test = np.array([[1.5, 1.5], [3.5, 3.5]])
y_pred = svm.predict(train_data[:5,:])

# 打印预测结果
print("预测结果：", y_pred)