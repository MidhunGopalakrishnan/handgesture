# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import glob
import cv2
import numpy as np
from scipy import spatial
import os
from pathlib import Path
from frameextractor import frameExtractor
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
    videos = glob.glob(os.path.join(inputPathName, "*"))
    frames_path = os.path.join( "frames_" + inputPathName)
    Path("frames_" + inputPathName).mkdir(parents=True, exist_ok=True)
    fileNumber = 0
    for video in videos:
        frameExtractor(video, frames_path, fileNumber)
        fileNumber += 1
    frames = glob.glob(os.path.join(frames_path, "*.png"))
    feature_vector = getFeatureVector(frames)
    np.savetxt(csvFileName, feature_vector, delimiter=",")


# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
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


training_data = np.genfromtxt(opfilename1, delimiter=",")
test_data = np.genfromtxt(opfilename2, delimiter=",")
res = []
for x in test_data:
    res.append(getGesture(x, test_data))
print(res)
# np.savetxt('Results_numbers_new.csv', res, delimiter=",", fmr="% d")
