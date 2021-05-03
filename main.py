import cv2
import numpy as np
from scipy import spatial
import os
from handshape_feature_extractor import HandShapeFeatureExtractor

np.savetxt('Results.csv', list(range(17))*3, fmt="%d")


def frameExtractor(videopath):
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no = int(video_length / 2)
    cap.set(1, frame_no)
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def generatePenultimateLayer(inputPathName):
    videos = []
    featureVectors = []
    for fileName in os.listdir(inputPathName):
        videos.append(os.path.join(inputPathName, fileName))
    for video in videos:
        frame = frameExtractor(video)
        feature = HandShapeFeatureExtractor.get_instance().extract_feature(frame)
        featureVectors.append(feature)
    return featureVectors


# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
trainVectors = generatePenultimateLayer("traindata")

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
testVectors = generatePenultimateLayer("test")


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

def getGesture(test_vector, train_penLayer):
    lst = []
    for x in train_penLayer:
        lst.append(spatial.distance.cosine(test_vector, x))
        gesture_num = lst.index(min(lst))
    return gesture_num


res = []
for x in testVectors:
    res.append(getGesture(x, trainVectors))
print(res)

