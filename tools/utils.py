import glob
import os
import random

def CreateMap(charFile):
    file = open(charFile)
    charTable = []
    charTable += file.read().split('\n')
    s2lDict = dict([(c, i) for i, c in enumerate(charTable)])
    l2sDict = dict([(i, c) for i, c in enumerate(charTable)])
    return  s2lDict, l2sDict

def SplitDataPath(dataDir, saveDir, sentTable, validPercent):
    pattern = f'{dataDir}/*.wav'
    sents = open(sentTable, 'r').read().split('\n')
    sents = list(filter(lambda x: x!='', sents))

    fileNames = []
    for path in glob.glob(pattern):
        tempName = path.split(os.sep)[-1]
        tempName = os.path.splitext(tempName)[0]
        fileNames.append(tempName)
    print(fileNames)
    trainNames = []
    testNames = []
    for sent in sents:
        curList = list(filter(lambda x: sent == x.split('_')[0], fileNames))
        curLen = len(curList)
        validNum = int(validPercent * curLen)
        print(f'validnum is: {validNum}')
        select = random.sample(range(0, curLen), validNum)
        for k in range(curLen):
            if k in select:
                testNames.append(curList[k])
            else:
                trainNames.append(curList[k])
    ###
    # 把路径保存到文件里
    with open(f'{saveDir}/train.txt', 'w') as f:
        for name in trainNames:
            f.write(name)
            f.write('\n')
    with open(f'{saveDir}/test.txt', 'w') as f:
        for name in testNames:
            f.write(name)
            f.write('\n')

def Label2Sent(tensor, l2sDict, netType, vocabType):

    sent = []
    sentLen = tensor.shape[0]
    for i in range(sentLen):
        sinTensor = tensor[i]
        sinSent = ''
        if netType == 'ed':
            pass
        if netType == 'ctc':
            pass
        if netType == 'conv':
            sinSent = ConvTrans(sinTensor, l2sDict)
        sent.append(sinSent)
    return sent

def CTCTrans(tensor, l2sDict):
    pass

def EDTrans(tensor, l2sDict):
    sent = ''
    for i in range(tensor.shape[0]):
        tranChar = l2sDict[tensor[i].item()]
        if len(tranChar) == 1:
            sent += tranChar
    return sent

def ConvTrans(tensor, l2sDict):
    sent = l2sDict[tensor.item()]
    return sent

def WordTrans(tensor, l2sDict):
    sent = ''
    maxLen = tensor.shape[0]
    for i in range(maxLen):
        tranWord = l2sDict[tensor[i].item()]
        if tranWord not in ['SOS', 'EOS']:
            if i != maxLen - 1:
                sent += tranWord
                sent += ' '
            else:
                sent += tranWord
    return sent