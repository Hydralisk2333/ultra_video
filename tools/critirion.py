from jiwer import wer
import numpy as np

def EvalWer(target, prediction):
    # target 是list类型的
    length = len(target)
    totalError = 0.0
    for i in range(length):
        error = wer(target[i], prediction[i])
        totalError += error
    return totalError