from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def CutSignalBiside(wavSignal, fs):
    leftOff = int(0.3 * fs)
    rightOff = int(0.2 * fs)
    wavLen = len(wavSignal)
    wavSignal = wavSignal[leftOff:wavLen-rightOff]
    return wavSignal

def CutSTFTBiside(Zxx):
    # leftOff = 32
    leftOff = 32
    rightOff = 12
    timeLen = Zxx.shape[1]
    Zxx = Zxx[:,leftOff:timeLen-rightOff]
    return Zxx

