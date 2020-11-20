import glob
import math
import os
import shutil
from itertools import repeat

import multiprocessing
from multiprocessing import Pool

from pre_data.video2pic import *


def GetPath(corpus, saveP, subPattern):
    # saveP = 'all_path.txt'
    # pattern = f'{corpus}{os.sep}volunteers{os.sep}*{os.sep}straightcam{os.sep}*.mp4'
    # subPattern 前面不能有斜杠这种东西
    # corpus = corpus.strip(os.sep)
    pattern = f'{corpus}{os.sep}{subPattern}'

    file = open(saveP, 'w')

    print(corpus)
    if os.path.exists(corpus):
        print(f"{corpus} can read")
    else:
        print('cant')

    for path in glob.glob(pattern):
        name = os.path.splitext(path)[0]
        name = name.split(f'{corpus}{os.sep}')[-1]
        name = name.replace(os.sep, '/')
        # prefix = f'{splitPath[-4]}/{splitPath[-3]}/{splitPath[-2]}/'
        # name = splitPath[-1].split('.')[0]
        name = name + '\n'
        file.write(name)
    file.close()

def GridContentChange(oriPath, desPath):
    inFile = open(oriPath, 'r')
    outFile = open(desPath, 'w')
    lines = inFile.read().split('\n')
    lines = list(filter(lambda x:x!='', lines))
    for sin in lines:
        sin = sin.replace('align', 'video')
        outFile.write(sin)
        outFile.write('\n')
    inFile.close()
    outFile.close()

def GetWordVocab(contentPath, savePath):
    file = open(contentPath, 'r')
    keyValue = file.read().split('\n')
    keyValue = list(filter(lambda x:x!='', keyValue))
    vocab = set()
    saveFile = open(savePath, 'w')
    for kv in keyValue:
        sent = kv.split(' ', 1)[-1]
        words = sent.split(' ')
        for word in words:
            vocab.add(word)
    saveFile.write(f'$\n')
    for word in vocab:
        saveFile.write(f'{word}\n')
    file.close()
    saveFile.close()

def GetPic(corpus, outputDir, dealP, predictorPath, videoFix):
    lines = open(dealP, 'r').read().split('\n')
    lines = list(filter(lambda x:x!='', lines))
    totalLen = len(lines)
    runNum = multiprocessing.cpu_count()
    # print(f'run num is : {runNum}')
    # perNum = totalLen // runNum
    perNum = math.ceil(totalLen / runNum)
    print(f'per num is : {perNum}')
    farg = []
    sarg = []
    for i in range(runNum):
        start = i * perNum
        end = (i + 1) * perNum
        if i != runNum - 1:
            farg.append(lines[start:end])
        else:
            farg.append(lines[start:])

    inArg = zip(farg, repeat(corpus), repeat(outputDir), repeat(predictorPath), repeat(videoFix))
    pool = Pool(runNum)
    pool.starmap(ParaPic, inArg)
    pool.close()
    pool.join()

def ParaPic(lines, corpus, outputDir, predictorPath, videoFix):
    for line in lines:
        path = line + '.' + videoFix
        path = path.replace('/', os.sep)
        fullP = os.path.join(corpus, path)
        discard, picDir = ExtractVideo(videoPath=fullP,
                                       corpusDir=corpus,
                                       outputDir=outputDir,
                                       predictorPath=predictorPath)
        print(f'{fullP} is end')

        if discard:
            file = open('log_msg.log', 'a+')
            shutil.rmtree(picDir)
            print(f'\n\n{path} discard\n\n')
            file.write(f'{path}\n')
            file.close()
