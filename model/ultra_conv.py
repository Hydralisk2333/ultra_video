import torch
import torch.nn as nn
import torchvision


class UltraConv(nn.Module):
    # (274, 48)
    def __init__(self, numClass):
        super(UltraConv, self).__init__()
        self.numClass = numClass

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5))
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dense1 = nn.Linear(in_features=6144, out_features=512)
        # self.dropout = nn.Dropout(0.5)
        self.relu = nn.PReLU()
        # self.relu = nn.Tanh()
        self.dense2 = nn.Linear(in_features=512, out_features=numClass)

    def forward(self, wavInput):
        wavOutput = self.conv1(wavInput)
        wavOutput = self.bn1(wavOutput)
        wavOutput = self.relu(wavOutput)
        wavOutput = self.pool1(wavOutput)
        # print(wavOutput)
        wavOutput = self.conv2(wavOutput)
        wavOutput = self.bn2(wavOutput)
        wavOutput = self.relu(wavOutput)
        wavOutput = self.pool2(wavOutput)
        # print(wavOutput)
        wavOutput = self.conv3(wavOutput)
        wavOutput = self.bn3(wavOutput)
        wavOutput = self.relu(wavOutput)
        wavOutput = self.pool3(wavOutput)

        wavOutput = wavOutput.view(wavOutput.shape[0], -1)

        # print(wavOutput.shape[1])
        wavOutput = self.dense1(wavOutput)
        # wavOutput = self.relu(wavOutput)
        return wavOutput

class ResNetUltra(torch.nn.Module):
    def __init__(self, classNum):
        super(ResNetUltra, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=512)
        print('cur conv is resnet18')

    def forward(self, x):
        # x shape = (batch, channel, w, h)
        x = self.resnet18(x)
        return x