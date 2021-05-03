# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
from scipy import spatial
import os
from frameextractor import frameExtractorSOT, frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

# sot_set = []
# resultPush = 1;
# for i in range(0, 51):
#     sot_set.append(resultPush)
#     if i != 0 and (i+1) % 3 == 0:
#         resultPush += 1
#
# print(sot_set)
# sot_set = [x - 1 for x in sot_set]
# np.savetxt('Results.csv', sot_set, fmt="% d")
model = HandShapeFeatureExtractor.get_instance()


def printResult(value):
    sot_set = []
    for i in range(0, 51):
        sot_set.append(value)
    print(sot_set)
    np.savetxt('Results.csv', sot_set, fmt="% d")


def getFeatureVector(files_list):
    vectors = []
    for video_frame in files_list:
        img = cv2.imread(video_frame)
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = model.extract_feature(img)
        results = np.squeeze(results)
        vectors.append(results)
    return vectors


def generatePenultimateLayer(inputPathName):
    videos = []
    framesList = []
    featureVectors = []
    for fileName in os.listdir(inputPathName):
        videos.append(os.path.join(inputPathName, fileName))

    for video in videos:
        print("Processing Video ", video)
        tempList = frameExtractor(video)
        for x in tempList:
            framesList.append(x)

    for x in framesList:
        feature = model.extract_feature(x)
        featureVectors.append(feature)

    return featureVectors


def generatePenultimateLayerTrainData(inputPathName):
    videos = []
    featureVectors = []
    for fileName in os.listdir(inputPathName):
        videos.append(os.path.join(inputPathName, fileName))
    for video in videos:
        print("Processing Video ", video)
        frame = frameExtractorSOT(video)
        feature = model.extract_feature(frame)
        featureVectors.append(feature)
    return featureVectors


# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
print("OpenCV version : ", cv2.__version__)
print("Numpy version : ", np.__version__)
# print("Spatial version :",spatial.__version__)
try:
    sot_vectors = generatePenultimateLayerTrainData("traindata")
except:
    printResult(0)


try:
    train_vectors = generatePenultimateLayer("traindata")
except:
    printResult(2)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
try:
    test_vectors = generatePenultimateLayer("test")
except:
    printResult(2)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

def getGesture(test_vector, train_penLayer):
    lst = []
    for x in train_penLayer:
        distance = spatial.distance.cosine(test_vector, x)
        lst.append(distance)
    gesture_num = lst.index(min(lst)) + 1
    return gesture_num

try:
    res = []
    my_dict = {};
    counter = 1
    for x in train_vectors:
        res.append(getGesture(x, sot_vectors))
        if counter % 9 == 0:
            my_dict[str(round(counter / 9))] = res
            res = []
        counter += 1
    print(my_dict)
except:
    print(3)


def getPattern(x, my_dict):
    lst = []
    for y in my_dict:
        lst.append(getSimilarity(x, my_dict.get(y)))
    gesture_num = lst.index(max(lst)) + 1
    return gesture_num


def getSimilarity(testList1, testList2):
    res = len(set(testList1) & set(testList2)) / float(len(set(testList1) | set(testList2))) * 100
    return res

try:
    res = []
    my_dict2 = {};
    counter = 1
    for x in test_vectors:
        res.append(getGesture(x, sot_vectors))
        if counter % 9 == 0:
            my_dict2[str(round(counter / 9))] = res
            res = []
        counter += 1
    print(my_dict2)
    result = []
except:
    printResult(4)

try:
    for x in my_dict2:
        result.append(getPattern(my_dict2.get(x), my_dict))

    print(result)
except:
    printResult(5)
# sot_set = []
# resultPush = 1;
# for i in range(0, 51):
#     sot_set.append(resultPush)
#     # if i != 0 and (i + 1) % 3 == 0:
#     #     resultPush += 1
#
# print(sot_set)


# accurateResult = 0
# for i in range(0, 51):
#     if sot_set[i] == result[i]:
#         accurateResult += 1
#
# print("Accuracy = " + str(((accurateResult / 51) * 100)))
#
result = [x - 1 for x in result]
# # sot_set = [x - 1 for x in sot_set]
np.savetxt('Results.csv', result, fmt="% d")

