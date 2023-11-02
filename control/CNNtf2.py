import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential


def load_data():
    on00all = np.zeros((13,200,9))
    train_data = np.zeros((26,200,9))
    for i in range(13):
        on00all[i,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on00_'+str(i+1)+'.txt')
    on45all = np.zeros((13,200,9))
    for i in range(13):
        train_data[i+13,:,:] = np.loadtxt('/home/nuci7/project/cf2/crazyflie-firmware/control/train_data/on45_'+str(i+1)+'.txt')

    print(np.shape(train_data))
    # for p in range(np.shape(train_data)[0]):
    #     train_data[p,:,4] = train_data[p,:,4] / train_data[p,:,3]
    #     train_data[p,:,5] = train_data[p,:,5] / train_data[p,:,3]
    #     train_data[p,:,6] = train_data[p,:,6] / train_data[p,:,3]
    # train_data = np.delete(train_data, [3,4,5,6,7], 1)
    train_data = tf.expand_dims(train_data,-1)
    train_data = np.delete(train_data, [3,7,8], 2)
    # print(np.shape(train_data[7,:,0,0]))
    # print(train_data[7,:,0,0])
    # plt.figure()
    # plt.plot(train_data[7,:,0,0])
    # plt.show()
    return train_data

def gen_label():
    label1 = np.zeros((26,2))
    label1[:13,0] = 1
    label1[13:,1] = 1
    return label1

class CNN(object):
    def __init__(self, train_x):
# def create_model(train_x):
        model = Sequential()
        model.add(tf.keras.layers.Conv2D(input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]),
                                    filters=2, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=1))
        model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=1))

        model.add(tf.keras.layers.Flatten())


        
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))

        

        # model.add(tf.keras.layers.Dense(32))
        # model.add(tf.keras.layers.Dense(16))
        # model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
        model.compile(loss='mse', optimizer='adam', metrics='accuracy')
        model.summary()
        self.model = model



# def create_dataset():
    





class Train():
    def __init__(self):
        # self.input1_train, self.input2_train, self.label1, self.label2 = create_dataset()
        self.input1_train = load_data()
        self.label1 = gen_label()
        self.model_usr1 = CNN(self.input1_train)
        # self.model_usr2 = CNN(self.input2_train)
    # create dataset
    # data_set, max_close_price, min_close_price = load_local_data(filenames)

    # train, test = split_data(data_set, ratio_train=0.8)
    

    # create model
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/Users/zhi/Project/upload_code/log_dir/tf_board_log')
    
    def train(self):
    # train model
        history_usr1 = self.model_usr1.model.fit(self.input1_train[3:24,:,:,:], self.label1, epochs=400, batch_size = 12, 
                            validation_split=0.25, verbose=1, shuffle=True)
        # history_usr2 = self.model_usr2.model.fit(self.input2_train, self.label2, epochs=1, batch_size = 1280,
        #                     validation_split=0.25, verbose=1, shuffle=True)


        os.makedirs('savedmodel', exist_ok=True)
        self.model_usr1.model.save('/home/nuci7/project/cf2/crazyflie-firmware/control/savedmodel/cnn_usr1_model.h5')
        print(history_usr1.history.keys())
        print(history_usr1.history['loss'],history_usr1.history['val_loss'])
        plt.figure(1)
        plt.plot(history_usr1.history['loss'])
        plt.plot(history_usr1.history['val_loss'])
        # plt.figure(2)
        # plt.plot(history_usr2.history['loss'])
        # plt.plot(history_usr2.history['val_loss'])
        # # plt.plot(history_usr2.history['loss'])s
        plt.grid(True)
        # plt.gca().set_ylim(0, 1)
        plt.show()

    # # save model
    # os.makedirs('savedmodel', exist_ok=True)
    # model_usr1.save('savedmodel/cnn_usr1_model.h5')

    # make predictions
    # raw_price = model.predict(test_x)
    # predict = recover_close_price(raw_price, max_close_price, min_close_price)
    # real = recover_close_price(test_y, max_close_price, min_close_price)

    # plot loss and prediction figure
    # plot_loss_prediction_figure(predict, real, history)





def test():
    SNR_db = [10]
    H1_real, H1_image = generate_channel(N, M, 0)
    H2_real, H2_image = generate_channel(N, M, 1)
    # for k in range(n_iteration):
    #     print('testing operation %d' % (k))

    #     for i in range(len(SNR_db)):
    #         SPC_test, test_label1, test_label2 = generate(M, N, batch_size * test_size)
    #         signal_power = np.mean(pow(SPC_test, 2))
    #         sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

    #         input1_test = tf.expand_dims(generate_input(H1_real, H1_image, SPC_test, N, batch_size * test_size, sigma),-1)
    #         input1_test = tf.reshape(input1_test, [-1, 4, 4, 1])

    #         input2_test = tf.expand_dims(generate_input(H2_real, H2_image, SPC_test, N, batch_size * test_size, sigma),-1)
    #         input2_test = tf.reshape(input2_test, [-1, 4, 4, 1])


    #         model_pre1 = tf.keras.models.load_model('savedmodel/cnn_usr1_model.h5')
    #         loss,accuracy = model_pre1.evaluate(input1_test, test_label1)
    #         print('\ntest loss',loss)
    #         print('accuracy',accuracy)
            
    i = 0
    SPC_test, test_label1, test_label2 = generate(M, N, 50)
    signal_power = np.mean(pow(SPC_test, 2))
    sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

    input1_test = tf.expand_dims(generate_input(H1_real, H1_image, SPC_test, N, 50, sigma),-1)
    input1_test = tf.reshape(input1_test, [-1, 4, 4, 1])

    input2_test = tf.expand_dims(generate_input(H2_real, H2_image, SPC_test, N, 50, sigma),-1)
    input2_test = tf.reshape(input2_test, [-1, 4, 4, 1])
    
    
    model_pre1 = tf.keras.models.load_model('savedmodel/cnn_usr1_model.h5')
    xxx,yyy = model_pre1.evaluate(input1_test, test_label1)
    print(xxx,yyy)
    # print('\ntest loss',metrics)
    # print('accuracy',metrics[1])



            # ERROR_user1[i, k] = 1 - ac1

            # input2_test = generate_input(H2_real, H2_image, SPC_test, N, batch_size * test_size, sigma)

            # ac2, out_test2 = sess.run([acc2, output2], feed_dict={in2: input2_test, la2: test_label2})


            # ERROR_user2[i, k] = 1 - ac2


def littletest():
    SNR_db = [10]
    H1_real, H1_image = generate_channel(N, M, 0)
    H2_real, H2_image = generate_channel(N, M, 1)            
    i = 0
    SPC_test, test_label1, test_label2 = generate(M, N, 50)
    signal_power = np.mean(pow(SPC_test, 2))
    sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

    input1_test = tf.expand_dims(generate_input(H1_real, H1_image, SPC_test, N, 50, sigma),-1)
    input1_test = tf.reshape(input1_test, [-1, 4, 4, 1])
    model_pre1 = tf.keras.models.load_model('/home/nuci7/project/cf2/crazyflie-firmware/control/savedmodel/cnn_usr1_model.h5')
    ooout = model_pre1.predict(input1_test)
    print(np.shape(test_label1))
    print(np.shape(ooout))

def test_n():
    model_pre1 = tf.keras.models.load_model('/home/nuci7/project/cf2/crazyflie-firmware/control/savedmodel/cnn_usr1_model.h5')
    ooout1 = model_pre1.predict(load_data()[0:2,:,:,:])
    ooout2 = model_pre1.predict(load_data()[24:26,:,:,:])

    # print(np.shape(test_label1))
    print(np.shape(ooout1))
    print(ooout1)
    print(ooout2)


def motor_ratio():
    all_data = load_data()
    for p in range(np.shape(all_data)[0]):
        all_data[p,:,4,:] = all_data[p,:,4,:] / all_data[p,:,3,:]
        all_data[p,:,5,:] = all_data[p,:,5,:] / all_data[p,:,3,:]
        all_data[p,:,6,:] = all_data[p,:,6,:] / all_data[p,:,3,:]




if __name__ == "__main__":
    # littletest()
    app = Train()
    app.train()
    
    # load_data()


    test_n()
    # load_data()
    # gen_label()
    # print(gen_label())
    # test()


# #########################################################################
# error1 = np.mean((ERROR_user1), axis=1)
# error2 = np.mean((ERROR_user2), axis=1)
# print(H1_real)
# plt.figure()
# plt.semilogy(SNR_db, error1, ls='--', marker='o', label='user1')
# plt.semilogy(SNR_db, error2, ls='--', marker='+', label='user2')
# plt.grid()
# plt.legend()
# plt.ylim(pow(10, -6), pow(10, 0))
# plt.xlabel('SNR')
# plt.ylabel('SER')
# plt.title('SER of user2 in 4X4 MIMO_NOMA BPSK_DNN')
# plt.savefig('SER_44MIMO_NOMA_DNN_BPSK')
# plt.show()

# print(error1)
# print(error2)

