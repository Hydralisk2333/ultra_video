import os
import pickle

import torch.utils.data as td

from pre_data.fun_wav import LoadUltra
from pre_data.load_label import Str2Label
from pre_data.load_video import LoadVid
from tools.utils import *


class MyDataset(td.Dataset):
    def __init__(self, corpusPath, pathFile, contentFile, vocabPath, limitPath,
                 vidMaxLen, ultraMaxLen, labMaxLen,
                 imgPostfix, imgShape, channelType,
                 ultraPostfix, sonicFreq, lipOffset, nfft, frameTime, aheadTime, winLen, syLen,
                 netType, vocabType, mode):
        self.corpusPath = corpusPath
        self.vidMaxLen = vidMaxLen
        self.ultraMaxLen = ultraMaxLen
        self.labMaxLen = labMaxLen

        self.imgPostfix = imgPostfix
        self.imgShape = imgShape
        self.channelType = channelType

        self.ultraPostfix = ultraPostfix
        self.sonicFreq = sonicFreq
        self.lipOffset = lipOffset
        self.nfft = nfft
        self.frameTime = frameTime
        self.aheadTime = aheadTime
        self.winLen = winLen
        self.syLen = syLen

        self.netType = netType
        self.vocabType = vocabType

        self.mode = mode

        self.s2lDict, self.l2sDict = CreateMap(vocabPath)

        with open(pathFile, 'r') as f:
            lines = f.read().split('\n')
            self.allPath = list(filter(lambda x:x, lines))

        with open(contentFile, 'r') as f:
            lines = f.read().split('\n')
            textPath = list(filter(lambda x: x, lines))
            self.textDict = {}
            for path in textPath:
                path = path.split(' ', 1)
                self.textDict[path[0]] = path[1]

        with open(limitPath, 'rb') as f:
            self.ultraMinValue, self.ultraMaxValue = pickle.load(f)

    def __len__(self):
        return len(self.allPath)

    def __getitem__(self, index):
        realPath = self.allPath[index]
        imgDir = os.path.join(self.corpusPath, 'lip_pic', realPath)
        ultraPath = os.path.join(self.corpusPath, 'ultra', f'{realPath}.{self.ultraPostfix}')
        ultraInput = []
        ultraLen = []
        videoInput = []
        videoLen = []
        labelInput= []
        labelLen= []
        if self.mode == 'ultra':
            ultraInput, ultraLen = LoadUltra(
                ultraMaxLen=self.ultraMaxLen,
                filePath=ultraPath,
                sonicFreq=self.sonicFreq,
                lipOffset=self.lipOffset,
                nfft=self.nfft,
                winLen=self.winLen,
                syLen=self.syLen,
                frameTime=self.frameTime,
                aheadTime=self.aheadTime
            )
            ultraInput = (ultraInput - self.ultraMinValue) / (self.ultraMaxValue - self.ultraMinValue)
        if self.mode == 'video':
            videoInput, videoLen = LoadVid(imgDir, self.vidMaxLen, self.imgPostfix, self.imgShape, self.channelType)
        if self.mode == 'dual':
            ultraInput, ultraLen = LoadUltra(
                ultraMaxLen=self.ultraMaxLen,
                filePath=ultraPath,
                sonicFreq=self.sonicFreq,
                lipOffset=self.lipOffset,
                nfft=self.nfft,
                winLen=self.winLen,
                syLen=self.syLen,
                frameTime=self.frameTime,
                aheadTime=self.aheadTime
            )
            ultraInput = (ultraInput - self.ultraMinValue) / (self.ultraMaxValue - self.ultraMinValue)
            videoInput, videoLen = LoadVid(imgDir, self.vidMaxLen, self.imgPostfix, self.imgShape, self.channelType)
        labelInput, labelLen = Str2Label(self.textDict[realPath], self.s2lDict, self.labMaxLen, self.netType, self.vocabType)
        # videoInput shape = (T, H, W, C)

        # print(f'video shape: {videoInput.shape}')
        # print(f'label shape: {labelInput.shape}')

        sample = {'ultraInput':[], 'videoInput':[], 'labelInput':[], 'ultraLen':[], 'videoLen':[], 'labelLen':[]}
        sample['ultraInput'] = ultraInput
        sample['ultraLen'] = ultraLen
        sample['videoInput'] = videoInput
        sample['videoLen'] = videoLen
        sample['labelInput'] = labelInput
        sample['labelLen'] = labelLen
        return sample
    # 最后输出的数据类型是torch