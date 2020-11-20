import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model.simed import *
from model.ultra_conv import *
from pre_data.data_loader import MyDataset
from pre_data.fun_wav import CalLimitValue
from tools.critirion import EvalWer
from tools.utils import *


class LipReading(nn.Module):
    def __init__(self, paras):
        super(LipReading, self).__init__()
        ## 模型固有参数
        self.corpusPath = paras['corpusPath']
        self.charPath = paras['charPath']
        self.trainPath = paras['trainPath']
        self.testPath = paras['testPath']
        self.contentPath = paras['contentPath']
        self.trainBatch = paras['trainBatch']
        self.testBatch = paras['testBatch']
        self.device = paras['device']
        self.limitPath = paras['limitPath']

        self.learnRate = paras['learnRate']
        self.clip = paras['clip']
        self.epoch = paras['epoch']
        self.saveEvery = paras['saveEvery']
        self.checkPath = paras['checkPath']
        self.testWer = 10000.0
        self.testLoss = 10000.0

        self.mode = paras['mode']
        ## 视频参数
        self.videoLen = paras['vidMaxLen']
        self.enInputSize = paras['enInputSize']
        self.enHiddenSize = paras['enHiddenSize']
        self.imgShape = paras['imgShape']
        self.numLayer = paras['numLayer']
        self.dropout = paras['dropout']
        self.batchFirst = paras['batchFirst']
        self.imgPostfix = paras['imgPostfix']
        self.channelType = paras['channelType']
        self.vidMaxLen = paras['vidMaxLen']
        ## 超声波参数
        self.ultraPostfix = paras['ultraPostfix']
        self.ultraMaxLen = paras['ultraMaxLen']
        self.sonicFreq = paras['sonicFreq']
        self.lipOffset = paras['lipOffset']
        self.nfft = paras['nfft']
        self.winLen = paras['winLen']
        self.syLen = paras['syLen']
        self.frameTime = paras['frameTime']
        self.aheadTime = paras['aheadTime']
        ## 标签参数
        self.labMaxLen = paras['labMaxLen']

        ## 其他公共参数
        self.netType = paras['netType']
        self.vocabType = paras['vocabType']

        print(self.device)

        self.change = False

        # 字符索引映射表
        self.s2lDict, self.l2sDict = CreateMap(self.charPath)
        self.classNum = len(self.s2lDict)

        # 定义网络模型
        self.videoM = Resnet3D(self.classNum)
        self.ultraM = ResNetUltra(self.classNum)
        outSize = 512
        if self.mode == 'dual':
            self.fc = nn.Linear(2*outSize, self.classNum)
        else:
            self.fc = nn.Linear(outSize, self.classNum)

        self.videoM = self.videoM.to(self.device)
        self.ultraM = self.ultraM.to(self.device)
        self.fc = self.fc.to(self.device)

        # 优化器
        self.videoOpt = optim.Adam(self.videoM.parameters())
        self.ultraOpt = optim.Adam(self.ultraM.parameters())
        self.fcOpt = optim.Adam(self.fc.parameters())

        # 获得根目录路径
        self.lastPath = ''

        if not os.path.exists(self.limitPath):
            CalLimitValue(corpus=self.corpusPath,
                          limitPath=self.limitPath,
                          sonicFreq=self.sonicFreq,
                          lipOffset=self.lipOffset,
                          nfft=self.nfft,
                          winLen=self.winLen,
                          syLen=self.syLen,
                          frameTime=self.frameTime,
                          aheadTime=self.aheadTime)
            print('limit value calculate over\n')


    def GetLoader(self, pathFile, batch):
        dataset = MyDataset(
            corpusPath=self.corpusPath,
            pathFile=pathFile,
            contentFile=self.contentPath,
            vocabPath=self.charPath,
            limitPath=self.limitPath,
            vidMaxLen=self.vidMaxLen,
            ultraMaxLen=self.ultraMaxLen,
            labMaxLen=self.labMaxLen,
            imgPostfix=self.imgPostfix,
            imgShape=self.imgShape,
            channelType=self.channelType,
            ultraPostfix=self.ultraPostfix,
            sonicFreq=self.sonicFreq,
            lipOffset=self.lipOffset,
            nfft=self.nfft,
            frameTime=self.frameTime,
            aheadTime=self.aheadTime,
            winLen=self.winLen,
            syLen=self.syLen,
            netType=self.netType,
            vocabType=self.vocabType,
            mode=self.mode
        )
        loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)
        return loader

    def DealData(self, loader, curEpoch=-1, lossFn=None):
        allLoss = 0
        allWer = 0
        count = 0
        for idx, sample in enumerate(loader):

            ultraInput = sample['ultraInput']
            ultraLen = sample['ultraLen']
            videoInput = sample['videoInput']
            videoLen = sample['videoLen']
            labelInput = sample['labelInput']
            labelLen = sample['labelLen']

            if self.mode == 'ultra':
                ultraInput = ultraInput.to(self.device)
                ultraInput = ultraInput.float()
                ultraInput = ultraInput.unsqueeze(1)
                ultraOutput = self.ultraM(ultraInput)
                # ultraOutput = (B, 512)
                content = ultraOutput
            if self.mode == 'video':
                videoInput = videoInput.to(self.device)
                videoInput = videoInput.float()
                videoInput = videoInput.permute(0, 4, 1, 2, 3).contiguous()
                videoOutput = self.videoM(videoInput)
                # videoOutput = (B, 1, 512)
                content = videoOutput
            if self.mode == 'dual':
                ultraInput = ultraInput.to(self.device)
                ultraInput = ultraInput.float()
                videoInput = videoInput.permute(0, 4, 1, 2, 3).contiguous()
                ultraInput = ultraInput.unsqueeze(1)
                ultraOutput = self.ultraM(ultraInput)

                videoInput = videoInput.to(self.device)
                videoInput = videoInput.float()
                videoOutput = self.videoM(videoInput)
                content = torch.cat((ultraOutput, videoOutput), dim=-1)

            output = self.fc(content)
            # 将数据放到device上
            labelInput = labelInput.to(self.device)
            labelInput = labelInput.long()
            labelInput = labelInput.squeeze(1)

            curCount = len(labelInput)
            count += curCount

            if curEpoch >= 0:

                self.ultraOpt.zero_grad()
                self.videoOpt.zero_grad()
                self.fcOpt.zero_grad()
                loss = lossFn(output, labelInput)

                loss.backward()
                # _ = nn.utils.clip_grad_norm_(self.ultraM.parameters(), self.clip)
                # _ = nn.utils.clip_grad_norm_(self.videoM.parameters(), self.clip)
                # _ = nn.utils.clip_grad_norm_(self.fc.parameters(), self.clip)
                if self.mode == 'ultra':
                    self.ultraOpt.step()
                if self.mode == 'video':
                    self.videoOpt.step()
                if self.mode == 'dual':
                    self.ultraOpt.step()
                    self.videoOpt.step()
                self.fcOpt.step()
                allLoss += loss * curCount
            else:
                loss = lossFn(output, labelInput)
                allLoss += loss * curCount

            predLabel = torch.max(output, dim=-1)[-1]
            predSent = Label2Sent(predLabel, self.l2sDict, self.netType, vocabType=self.vocabType)
            targetSent = Label2Sent(labelInput, self.l2sDict, self.netType, vocabType=self.vocabType)

            curWer = EvalWer(targetSent, predSent)
            allWer += curWer

            if idx == len(loader) - 1:
                sentLen = len(targetSent)
                for i in range(sentLen):
                    print(f'prediction: {predSent[i]} || target: {targetSent[i]}')

        if curEpoch >= 0:
            printMsg = 'epoch {}, loss is {:.4f}, wer is {:.4f}\n\n'.format(curEpoch, allLoss / count, allWer / count)
        else:
            testWer = allWer / count
            testLoss = allLoss / count
            printMsg = 'test loss is {:.4f}, wer is {:.4f}\n\n'.format(allLoss / count, allWer / count)
            if testLoss <= self.testLoss:
                if self.testWer > testWer:
                    self.testWer = testWer
                    self.testLoss = testLoss
                    self.change = True
                    self.SaveCheck(self.testLoss, self.testWer)

        print(printMsg)

        if curEpoch >= 0:
            lossPath = 'train_loss.txt'
            with open(lossPath, 'a+') as f:
                msg = 'epoch {}, loss is {:.4f}, wer is {:.4f}\n'.format(curEpoch, allLoss / count, allWer / count)
                f.write(msg)


    def Train(self, teachRate):
        self.startEpoch = 0
        self.videoM.train()
        self.ultraM.train()
        self.fc.train()
        torch.cuda.empty_cache()

        lossFn = nn.CrossEntropyLoss()
        if teachRate == 1:
            searchType = 'teach'
        else:
            searchType = 'greedy'

        if self.checkPath!=None:
            self.LoadCheck(self.checkPath)

        trainLoader = self.GetLoader(self.trainPath, self.trainBatch)

        for t in range(self.startEpoch, self.epoch):
            self.change = False
            print(f'current epoch is {t}')
            print('train process')

            self.DealData(trainLoader, t, lossFn)
            # 顺便做测试
            with torch.no_grad():
                self.Test()

            if self.change:
                with open('testWer.txt', 'a+') as f:
                    f.write(f'test wer is {self.testWer}, cur epoch is {t}\n')
            # 判断是否收敛，收敛就退出
            # 存储checkpoint
            # realT = t+1
            # if realT % self.saveEvery == 0:
            #     self.SaveCheck(realT)

    def Test(self):
        self.videoM.train()
        self.ultraM.train()
        self.fc.train()
        torch.cuda.empty_cache()

        searchType = 'greedy'
        lossFn = nn.CrossEntropyLoss()

        testLoader = self.GetLoader(self.testPath, self.testBatch)
        print('test process')
        # 取出数据，训练
        self.DealData(testLoader, lossFn=lossFn)

    def SaveCheck(self, testLoss, testWer):
        directory = f'weights_{self.mode}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        savePath = os.path.join(directory, 'loss{:.4f}_wer{:.4f}.tar'.format(testLoss, testWer))
        torch.save({
            'ultraM': self.ultraM.state_dict(),
            'videoM': self.videoM.state_dict(),
            'fc': self.fc.state_dict(),
            'ultraOpt': self.ultraOpt.state_dict(),
            'videoOpt': self.videoOpt.state_dict(),
            'fcOpt': self.fcOpt.state_dict(),
            'testWer': self.testWer
        }, savePath)
        # if self.lastPath != '':
        #     os.remove(self.lastPath)
        # self.lastPath = savePath

    def LoadCheck(self, checkPath=None):
        if checkPath != None:
            checkpoint = torch.load(checkPath)
            self.model.load_state_dict(checkpoint['model'])
            self.opt.load_state_dict(checkpoint['opt'])
            self.testWer = checkpoint['testWer']
            self.lastPath = checkPath