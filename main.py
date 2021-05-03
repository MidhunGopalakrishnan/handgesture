# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
from scipy import spatial
import os
from handshape_feature_extractor import HandShapeFeatureExtractor

model = HandShapeFeatureExtractor.get_instance()


def frameExtractorSOT(videopath):
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no = int(video_length / 1.5)
    cap.set(1, frame_no)
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def generatePenultimateLayer(inputPathName):
    videos = []
    featureVectors = []
    for fileName in os.listdir(inputPathName):
        if fileName.endswith(".mp4"):
            videos.append(os.path.join(inputPathName, fileName))

    for video in videos:
        print("Processing Video ", video)
        cap = cv2.VideoCapture(video)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if video_length > 20:
            for pct in range(1, 20):
                frame_no = round((video_length * pct) / 20);
                cap.set(1, frame_no)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                feature = model.extract_feature(frame)
                featureVectors.append(feature)
    return featureVectors


def generatePenultimateLayerTrainData(inputPathName):
    videos = []
    featureVectors = []
    for fileName in os.listdir(inputPathName):
        if fileName.endswith(".mp4"):
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
sot_vectors = generatePenultimateLayerTrainData("traindata")
train_vectors = generatePenultimateLayer("traindata")

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
test_vectors = generatePenultimateLayer("test")


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


def getPattern(x, my_dict):
    lst = []
    for y in my_dict:
        lst.append(getSimilarity(x, my_dict.get(y)))
    gesture_num = lst.index(max(lst)) + 1
    return gesture_num


def getSimilarity(testList1, testList2):
    res = len(set(testList1) & set(testList2)) / float(len(set(testList1) | set(testList2))) * 100
    return res


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

for x in my_dict2:
    result.append(getPattern(my_dict2.get(x), my_dict))

result = [x - 1 for x in result]
print(result)
np.savetxt('Results.csv', result, fmt="%d")
