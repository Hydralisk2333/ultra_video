import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import math
import numpy as np
import pickle

class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.fc = nn.Linear(in_features=512, out_features=512)

    def forward(self, x):
        batch = x.shape[0]
        time = x.shape[1]
        channel = x.shape[2]
        w = x.shape[3]
        h = x.shape[4]
        x = x.view(-1, channel, w, h)
        x = self.resnet18(x)
        x = x.view(batch, time, x.shape[-1])
        return x

class VideoModel(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayer, batchFirst, dropout):
        super(VideoModel, self).__init__()
        self.resnet = ResNet()
        self.lstm = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayer, batch_first=batchFirst, dropout=dropout)

    def forward(self, eInput):
        output = eInput
        output = self.resnet(output)
        output, (h, s) = self.lstm(output)

        return output[:, -1, :]

class Resnet3D(nn.Module):
    def __init__(self, numClass):
        super(Resnet3D, self).__init__()
        self.resnet3D = torchvision.models.video.r3d_18(num_classes=512)

    def forward(self, inputData):
        outputData = inputData
        outputData = self.resnet3D(outputData)
        return outputData