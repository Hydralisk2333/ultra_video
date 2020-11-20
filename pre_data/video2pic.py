import math
import os

import cv2
import dlib
import skvideo.io
import numpy as np

def ExtractVideo(videoPath, corpusDir, outputDir, predictorPath):

    dirName = 'Log'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    logName = 'log/video_log.log'

    corpusDir = corpusDir.strip(os.sep)
    corpusDir += os.sep

    singleDir = videoPath.split(corpusDir)[-1]
    singleDir = os.path.splitext(singleDir)[0]

    mouthDesPath = os.path.join(outputDir, singleDir)
    if not os.path.exists(mouthDesPath):
        os.makedirs(mouthDesPath)

    inputparameters = {}
    outputparameters = {}
    reader = skvideo.io.FFmpegReader(videoPath,
                                     inputdict=inputparameters,
                                     outputdict=outputparameters)
    # 得到视频的图像shape
    video_shape = reader.getShape()

    counter = 1
    # 确定最大取得帧数

    pics = []
    for frame in reader:
        shape = frame.shape[0:2]
        # 如果视频过大，需要减小尺寸，720p以上缩放
        if shape[0] >= 720 and shape[1] >= 720:
            shape = (shape[1] // 3, shape[0] // 3)
            frame = cv2.resize(frame, shape)
        pics.append(frame)
    pics = np.array(pics)

    discard = DealFrame(pics, counter, mouthDesPath, predictorPath)

    return discard, mouthDesPath


def DealFrame(pics, counter, mouthDP, predictorPath):

    discard = False
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictorPath)

    width_crop_max = 0
    height_crop_max = 0

    (num_frames, h, w, c) = pics.shape

    for frame in pics:
        #print('frame_shape:', frame.shape)

        # 检测每一个帧,返回多个脸的左上，右下坐标
        detections = detector(frame, 1)
        # 提取唇部的最后20个特征点
        marks = np.zeros((2, 20))
        # 所有未归一化的脸部特征

        # 如果检测到脸部，输出检测到脸部的个数
        #print(len(detections))
        if len(detections) > 0:
            # k是表示第几张脸，d表示对应脸所在的左上和右下位置
            for k, d in enumerate(detections):
                # 脸部的shape，返回68个关键点
                shape = predictor(frame, d)
                # co一个中间结点计数值，用来计数当前选了几个和嘴唇相关点
                co = 0
                # 找出嘴唇
                for ii in range(48, 68):
                    """
                    循环用于提取和嘴唇相关的特征
                    """
                    # 获取对应的特征点
                    X = shape.part(ii)
                    A = (X.x, X.y)
                    marks[0, co] = X.x
                    marks[1, co] = X.y
                    co += 1

                # 获取截取嘴唇框图的极限值（top-left & bottom-right）
                # X_left表示图像左边，Y_left表示图像下部，X_right表示图像右端，Y_right表示图像上端
                X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                    int(np.amax(marks, axis=1)[0]), int(np.amax(marks, axis=1)[1])]

                # 得到嘴唇的中心
                X_center = (X_left + X_right) / 2.0
                Y_center = (Y_left + Y_right) / 2.0

                # 为裁剪设置一个边界
                # border是设置的边界值
                # if h >= 720 and w >= 720:
                #     border = 40
                # else:
                #     border = 30
                border = min(h, w) // 12

                X_left_new = X_left - border
                Y_left_new = Y_left - border
                X_right_new = X_right + border
                Y_right_new = Y_right + border

                # 用于裁剪的框图长宽
                width_new = X_right_new - X_left_new
                height_new = Y_right_new - Y_left_new
                width_current = X_right - X_left
                height_current = Y_right - Y_left

                # 确定裁剪矩阵维度（主要任务是生成自适应的区域）
                if width_crop_max == 0 and height_crop_max ==0:
                    width_crop_max = width_new
                    height_crop_max = height_new
                else:
                    width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                    height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)

                # 得到裁剪框的大小
                X_left_crop = int(X_center - width_crop_max / 2.0)
                X_right_crop = int(X_center + width_crop_max / 2.0)
                Y_left_crop = int(Y_center - height_crop_max / 2.0)
                Y_right_crop = int(Y_center + height_crop_max / 2.0)

                if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                    # 保存嘴部区域
                    # 注意cv中的图像是先算高度，再算宽度
                    mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]
                    mouth = mouth[...,::-1]

                    # mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                    # print("gray shape:", mouth.shape)
                    # mouth = cv2.resize(mouth, (FRAME_COLS, FRAME_ROWS))
                    last_mouth = mouth
                    #下面一行用来控制输出裁切的图片
                    cv2.imwrite(mouthDP + '/' + str(counter) + '.png', mouth)
                    #print("检测到可裁剪的嘴唇")

                else:
                    discard = True
                    print("no lip at", counter)

        else:
            discard = True
            print("没检测到嘴唇")
            print("no lip at", counter)
        counter += 1
    return discard