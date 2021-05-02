# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import glob
import shutil
import sys
import cv2
import numpy as np
from scipy import spatial
import os
from pathlib import Path
from frameextractor import frameExtractorSOT, frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor


def getFeatureVector(files_list):
    model = HandShapeFeatureExtractor.get_instance()
    vectors = []
    for video_frame in files_list:
        img = cv2.imread(video_frame)
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = model.extract_feature(img)
        results = np.squeeze(results)
        vectors.append(results)
    return vectors


def generatePenultimateLayer(inputPathName, csvFileName):
    videos = []
    for fileName in os.listdir(inputPathName):
        videos.append(os.path.join(inputPathName, fileName))
    # videos = glob.glob(os.path.join(inputPathName, "*"))
    frames_path = os.path.join("frames_" + inputPathName)
    Path("frames_" + inputPathName).mkdir(parents=True, exist_ok=True)
    fileNumber = 0
    for video in videos:
        print("Processing Video ", video)
        frameExtractor(video, frames_path, fileNumber)
        fileNumber += 1

    frames = []
    for fileName in os.listdir(frames_path):
        if fileName.endswith(".png"):
            frames.append(os.path.join(frames_path, fileName))

    # frames = glob.glob(os.path.join(frames_path, "*.png"))
    feature_vector = getFeatureVector(frames)
    np.savetxt(csvFileName, feature_vector, delimiter=",")


def generatePenultimateLayerTrainData(inputPathName, csvFileName):
    videos = []
    for fileName in os.listdir(inputPathName):
        videos.append(os.path.join(inputPathName, fileName))
    # videos = glob.glob(os.path.join(inputPathName, "*"))
    frames_path = os.path.join("frames_sot_" + inputPathName)
    Path("frames_sot_" + inputPathName).mkdir(parents=True, exist_ok=True)
    fileNumber = 0
    for video in videos:
        print("Processing Video ", video)
        frameExtractorSOT(video, frames_path, fileNumber)
        fileNumber += 1
    frames = []
    for fileName in os.listdir(frames_path):
        if fileName.endswith(".png"):
            frames.append(os.path.join(frames_path, fileName))
    # frames = glob.glob(os.path.join(frames_path, "*.png"))
    feature_vector = getFeatureVector(frames)
    np.savetxt(csvFileName, feature_vector, delimiter=",")


# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
opfilename3 = 'train_sot_penultimate_layer.csv'
generatePenultimateLayerTrainData("traindata", opfilename3)

opfilename1 = 'training_penultimate_layer.csv'
generatePenultimateLayer("traindata", opfilename1)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
opfilename2 = 'testing_penultimate_layer.csv'
generatePenultimateLayer("test", opfilename2)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

def getGesture(test_vector, train_penLayer):
    lst = []
    for x in train_penLayer:
        lst.append(spatial.distance.cosine(test_vector, x))
    gesture_num = lst.index(min(lst)) + 1
    return gesture_num


training_data = np.genfromtxt(opfilename3, delimiter=",")
test_data = np.genfromtxt(opfilename1, delimiter=",")
res = []
my_dict = {};
counter = 1
for x in test_data:
    res.append(getGesture(x, training_data))
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


training_data = np.genfromtxt(opfilename3, delimiter=",")
test_data = np.genfromtxt(opfilename2, delimiter=",")
res = []
my_dict2 = {};
counter = 1
for x in test_data:
    res.append(getGesture(x, training_data))
    if counter % 9 == 0:
        my_dict2[str(round(counter / 9))] = res
        res = []
    counter += 1
print(my_dict2)
result = []


for x in my_dict2:
    result.append(getPattern(my_dict2.get(x), my_dict))

print(result)

sot_set = []
resultPush = 1;
for i in range(0, 51):
    sot_set.append(resultPush)
    if i != 0 and (i+1) % 3 == 0:
        resultPush += 1

print(sot_set)

accurateResult = 0
for i in range(0, 51):
    if sot_set[i] == result[i]:
        accurateResult += 1

print("Accuracy = " + str(((accurateResult / 51) * 100)))

sot_set = [x - 1 for x in sot_set]
np.savetxt('Results.csv', sot_set, fmt="% d")
