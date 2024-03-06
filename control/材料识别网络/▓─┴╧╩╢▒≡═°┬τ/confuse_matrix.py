import os
import json
import pandas as pd
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from prettytable import PrettyTable


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int,labels: list, model):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.model = model

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[int(t), int(p)] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels(%)')
        plt.ylabel('Predicted Labels(%)')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig('confusion-matrix.jpg', dpi=1000)  # 指定分辨率保存
        plt.show()

    def plot_probability(self):
        matrix = self.matrix
        print(matrix)

        BIGGER_SIZE = 7
        plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # 将数值变为概率！！！一般画图都需要概率信息
        matrix_sum = np.sum(matrix, axis=0)
        matrix_probability = np.zeros((self.num_classes, self.num_classes))
        print(matrix_sum)
        for i in range(self.num_classes):
            matrix_probability[i] = matrix[:, i] / matrix_sum[i]
        print(matrix_probability)

        plt.imshow(np.transpose(matrix_probability), cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=60)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        # plt.colorbar()
        # plt.xlabel('True Labels(%)')
        # plt.ylabel('Predicted Labels(%)')
        # plt.title('Confusion matrix')
        # 保存数据，导出
        num = len(range(self.num_classes))
        matrix_data = np.zeros((num,num))

        # 在图中标注数量/概率信息
        thresh = matrix_probability.max() / 2
        for y in range(self.num_classes):
            for x in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = matrix_probability[y, x]
                info_str = "%.1f" % ((matrix_probability[y, x]) * 100)
                if info_str == "-0.0":
                    info_str = "0.0"
                plt.text(y, x, info_str,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black", fontsize=5)
                matrix_data[y,x] = info_str
        dataframe = pd.DataFrame(matrix_data)
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(self.model+'-matrix_data.csv')
        plt.tight_layout()
        plt.savefig(self.model+'-confusion-matrix.jpg', dpi=1500)  # 指定分辨率保存
        plt.show()



