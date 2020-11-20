import os
import numpy as np
import cv2


def LoadVid(pDir, maxLen, imgPostfix, shape, channelType):
    # print(f'now load {pDir}')

    flag = 0
    if channelType == 3:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE

    files = os.listdir(pDir)
    files = list(filter(lambda file: file.find(imgPostfix) != -1, files))
    files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))

    array = [cv2.imread(os.path.join(pDir, file), flag) for file in files]
    array = list(filter(lambda im: not im is None, array))
    array = [cv2.resize(im, shape, interpolation=cv2.INTER_LANCZOS4) for im in array]
    array = np.stack(array, axis=0).astype(np.float32)

    if channelType > 3:
        picLen = array.shape[0]
        newLen = picLen - channelType + 1
        tempArray = np.zeros((newLen, channelType) + array.shape[1:])
        for i in range(newLen):
            tempArray[i,:,:,:] = array[i:i+channelType,:,:]
        array = tempArray
        array = array.transpose((0, 1, 2, 3))

    picLen1 = array.shape[0]
    picShape = array.shape
    picShape = (maxLen-picLen1,) + picShape[1:]
    pad = np.zeros(picShape, dtype=np.float32)
    array = np.concatenate((array, pad), axis=0)
    # 归一化
    array = array / 255.0
    return array, picLen1
