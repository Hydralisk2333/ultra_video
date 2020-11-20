import glob
import pickle

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import math
from pre_data.cut_wav import *
import cv2
import librosa.core
from collections import Counter

sonicFreq = 19000
nfft = 8192
lipOffset = 40

def FiltWav(signalWav, N, lowPass, highPass):
    b, a = signal.butter(N=N, Wn=[lowPass, highPass], btype='bandpass')
    filtSignal = signal.filtfilt(b, a, signalWav)  # signalWav为要过滤的信号
    return filtSignal

def SingleSTFT(filePath, sonicFreq, lipOffset, nfft, frameTime=0.15, aheadTime=0.01):
    fs, signalWav = wav.read(filePath)

    lowPass = 2 * (sonicFreq - lipOffset) / fs
    highPass = 2 * (sonicFreq + lipOffset) / fs

    filtSignal = FiltWav(signalWav=signalWav, N=2, lowPass=lowPass, highPass=highPass)

    lmsSignal = filtSignal
    nperseg = int(fs * frameTime)
    noverlap = int(fs * (frameTime-aheadTime))
    f, t, Zxx = signal.stft(lmsSignal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Zxx = abs(Zxx)

    Zxx = CutSTFTBiside(Zxx)
    t = t[0:Zxx.shape[1]]
    down, up = LipGraphPara(sonicFreq, lipOffset, fs, nfft)
    f = f[down:up]
    Zxx = Zxx[down:up,:]

    return t, f, Zxx

def LipGraphPara(sonicFreq, lipOffset, fs, nfft, padNum=5):
    span = fs / nfft
    basicPos = int(sonicFreq / fs * nfft)
    freqOffset = int(lipOffset // span) + padNum
    down = basicPos - freqOffset
    up = basicPos + freqOffset
    return down, up

def CalMean(Zxx):
    mean = np.mean(Zxx, axis=1)
    xShape = Zxx.shape[0]

    for i in range(xShape):
        mask = (Zxx[i, :] > mean[i])
        mask = mask.astype('int64')
        # print(mask.shape)
        # print(Zxx[i,:].shape)
        Zxx[i, :] = Zxx[i, :] * mask
    return Zxx

def SubAdjacent(Zxx):
    time = Zxx.shape[1]
    shape = Zxx.shape
    result = np.zeros((shape[0], shape[1]-1))
    for i in range(1, time):
        result[:, i-1] = (Zxx[:,i] - Zxx[:,i-1])
    return result

def GaussWeight(x, sigma):
    w = 1 / ((2 * np.pi) ** 0.5) * np.exp(-x*x/2/sigma/sigma)
    return w

def GaussSmooth(var, lenNum, sigma):
    varNum = len(var)
    varBefore = var[lenNum-1::-1]
    dataAfter = var[sigma-lenNum:]
    varAfter = dataAfter[::-1]
    var = np.concatenate((varBefore, var, varAfter))
    mid = lenNum//2
    coreIndex = [i - mid for i in range(lenNum)]
    coreIndex = np.array(coreIndex)
    w = GaussWeight(coreIndex, sigma)
    result = []
    # for i in range(mid, varNum-mid):
    #     temp = w*var[i-mid, i+]

    for i in range(lenNum, varNum+lenNum):
        temp = np.dot(w,var[i-mid:i+lenNum-mid])
        result.append(temp)
    return np.array(result)

def GaussSmooth2D(Zxx, kernel, sigma):
    Zxx = cv2.GaussianBlur(Zxx, kernel, sigma)
    return Zxx

def CheckActive(Zxx, offset=2):
    time = Zxx.shape[1]
    freq = Zxx.shape[0]

    noisePercent = 0.1

    activation = np.zeros((time,))
    noise = np.mean(Zxx, axis=1)
    mid = freq // 2
    offset = offset
    tempSum = np.sum(noise) - np.sum(noise[mid-1:mid-1+2*offset])
    noise = tempSum / (freq - 2*offset)

    if noise < 0:
        noise = noise * (1-noisePercent)
    else:
        noise = noise * (1+noisePercent)

    for i in range(time):
        curFreq = Zxx[:, i]
        # print(curFreq)
        tempSum = 0.0
        for j in range(freq):
            if j>=mid-offset+1 and j<=mid+offset:
                continue
            tempSum += curFreq[j]
        mean = tempSum / (freq - 2*offset)
        if mean > noise:
            activation[i] = 1
    return activation

def CheckGradActive(Zxx):
    time = Zxx.shape[1]

    sumRes = np.sum(Zxx, axis=0)

    sumRes = abs(sumRes)
    lenNum = 20
    sigma = 4
    sumRes = GaussSmooth(sumRes, lenNum, sigma)

    activation = np.zeros((time,))

    winLen = 8
    maxDiff = 30
    lowThresh = 50

    for i in range(0, time - winLen):
        curWin = sumRes[i:i+winLen]
        curMean = np.mean(curWin)
        curMax = np.max(curWin)
        if curMax - curMean > maxDiff or curMean > lowThresh:
            activation[i:i+winLen] = 1

    # plt.plot(sumRes)
    # plt.plot(activation*20)
    # plt.show()

    return activation

def SplitWord(activation, winLen):
    time = activation.shape[0]
    upBound = 2
    deCount = upBound
    start = -1
    index = []
    for i in range(time-winLen):
        window = activation[i:i+winLen]
        judge = (window == 0)
        # print(judge)
        if False not in judge:
            deCount += 1
            if deCount == 2:
                end = min(i + winLen - upBound, time-1)
                index.append((start, end))
                start = -1
            deCount = min(deCount, upBound)
        else:
            if deCount >= upBound:
                start = i
            deCount = 0
    if start != -1:
        index.append((start, time))
    return index

def WashWav(activation, index, syLen):
    for case in index:
        clip = activation[case[0]:case[1]]
        cal = Counter(clip)
        if cal[1] < syLen:
            activation[case[0]:case[1]] = 0
            index.remove(case)
    return activation, index

def GetUseful(filePath, sonicFreq, lipOffset, nfft, winLen, syLen, frameTime=0.15, aheadTime=0.01):

    data = SingleSTFT(
        filePath=filePath,
        sonicFreq=sonicFreq,
        lipOffset=lipOffset,
        nfft=nfft,
        frameTime=frameTime,
        aheadTime=aheadTime
    )
    t = data[0]
    f = data[1]
    Zxx = data[2]

    # Zxx = np.log10(Zxx)
    # activation = CheckActive(Zxx, offset=2)

    Zxx = SubAdjacent(Zxx)
    # tempZxx = Zxx ** 2
    # activation = CheckGradActive(tempZxx)
    # Zxx = tempZxx
    #
    # index = SplitWord(activation, winLen)
    #
    # activation, index = WashWav(activation, index, syLen)

    res = [Zxx]
    return res

def LoadUltra(ultraMaxLen, filePath, sonicFreq, lipOffset, nfft, winLen, syLen, frameTime=0.15, aheadTime=0.01):
    res = GetUseful(filePath, sonicFreq, lipOffset, nfft, winLen, syLen, frameTime, aheadTime)
    res = res[0]

    freq = res.shape[0]
    # middle = freq // 2
    # cutOffset = 3
    # res = np.concatenate((res[0:middle - cutOffset, :], res[middle + cutOffset:, :]), axis=0)
    res = GaussSmooth2D(res, (5, 5), 2)

    # maxValue = np.max(res)
    # minValue = np.min(res)
    # res = (res - minValue) / (maxValue - minValue)
    # print(res.shape)
    res = res.transpose(1, 0)
    resLen = res.shape[0]
    padLen = ultraMaxLen - resLen
    beforePadLen = padLen // 2
    afterPadLen = padLen - beforePadLen
    beforePad = np.zeros((beforePadLen, res.shape[1]))
    afterPad = np.zeros((afterPadLen, res.shape[1]))
    res = np.concatenate((res, beforePad, afterPad), axis=0)
    return res, resLen

def CalLimitValue(corpus, limitPath, sonicFreq, lipOffset, nfft, winLen, syLen, frameTime=0.15, aheadTime=0.01):
    pattern = 'ultra/*.wav'
    dirPath = os.path.join(corpus, pattern)
    allMin = 100000
    allMax = -100000
    for path in glob.glob(dirPath):
        ultraMaxLen = 300
        res, resLen = LoadUltra(ultraMaxLen, path, sonicFreq, lipOffset, nfft, winLen, syLen, frameTime, aheadTime)
        data = res[:resLen]
        maxValue = np.max(data)
        minValue = np.min(data)
        if maxValue > allMax:
            allMax = maxValue
        if minValue < allMin:
            allMin = minValue
    with open(limitPath, 'wb') as f:
        pickle.dump((allMin, allMax), f)

