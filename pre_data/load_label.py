import numpy as np

def Str2Label(sent, s2lDict, maxLen, netType, vocabType):

    if vocabType == 'char':
        sLen = len(sent)
        sent = sent.lower()
        if netType == 'ctc':
            res = [s2lDict[' ']] * maxLen
            for i in range(sLen):
                res[i] = s2lDict[sent[i]]
        elif netType == 'ed':
            res = [s2lDict['EOS']] * maxLen
            res[0] = s2lDict['SOS']
            for i in range(sLen):
                res[i + 1] = s2lDict[sent[i]]
        else:
            print('net type is wrong')
            res = []
    else:
        words = sent.split(' ')
        sLen = len(words)
        # res = [s2lDict['EOS']] * maxLen
        # res[0] = s2lDict['SOS']
        res = [0]*maxLen
        for i in range(sLen):
            res[i] = s2lDict[words[i]]

    res = np.array(res)
    return  res, sLen