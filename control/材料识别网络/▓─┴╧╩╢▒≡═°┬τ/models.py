#!/usr/bin/python
# -*-coding: utf-8 -*-
# author: SSR-ZenanLin
# data: 2022年1月6日
# description: 多种时间序列模型重构
import numpy as np
import torch
from torch import nn
from torchsummary import summary
# from torchkeras import summary
from torch.utils.tensorboard import SummaryWriter




class Conv1dNet(nn.Module):
    def __init__(self, inplanes=19, num_classes=12):
        super(Conv1dNet, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, 4, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(4)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # (-1, 64, 500)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)  # (-1, 64, 250)
        x = self.conv2(x)  # (-1, 128, 125)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)  # (-1, 128, 63)
        x = self.conv3(x)  # (-1, 256, 32)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)  # (-1, 256, 16)

        x = self.avgpool(x)  # (-1, 256, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def print_1D_CNN_structure(inplanes=19, num_classes=12):
    # 输出每层网络参数信息
    # 模型建立
    model = Conv1dNet(inplanes, num_classes)
    # 实例化SummaryWriter对象
    summary(model, (inplanes, 1600), batch_size=1, device="cpu")
    model = model.to("cpu")


def print_ResCNN_net_structure(inplanes=2, num_classes=4):
    # 输出每层网络参数信息
    # 模型建立
    model = ResCNN_Net(inplanes, num_classes)
    summary(model, (inplanes, 1600), batch_size=1, device="cpu")
    model = model.to("cpu")
#


def print_FR_CNN_net_structure(inplanes=5, num_classes=8):
    # 输出每层网络参数信息
    # 模型建立
    model = FR_CNN_Net(inplanes, num_classes)
    summary(model, (inplanes, 1600), batch_size=1, device="cpu")
    model = model.to("cpu")



def print_InceptionTime_net_structure(inplanes=5, num_classes=8):
    # 输出每层网络参数信息
    # 模型建立
    model = InceptionTime_Net(inplanes, num_classes)
    summary(model, (inplanes, 1600), batch_size=1, device="cpu")
    model = model.to("cpu")


def print_LSTM_FCN_net_structure(inplanes=2, num_classes=4):
    # 输出每层网络参数信息
    # 模型建立
    model = LSTM_FCN_Net(inplanes, 1600, num_classes)
    summary(model, (inplanes, 1600), batch_size=1, device="cpu")
    model = model.to("cpu")


# def print_net_structure(model_select="1D-CNN", inplane=2, num_classes=4):
#     if model_select == "1D-CNN":
#         print_1D_CNN_structure(inplane, num_classes)
#     elif model_select == "ResCNN":
#         print_ResCNN_net_structure(inplane, num_classes)
#     elif model_select == "LSTM-FCN":
#         print_LSTM_FCN_net_structure(inplane, num_classes)
#     elif model_select == "InceptionTime":
#         print_InceptionTime_net_structure(inplane, num_classes)
#     else:
#         print_1D_CNN_structure(inplane, num_classes)
