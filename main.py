import cv2
import numpy as np
from scipy import spatial
import os
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

model = HandShapeFeatureExtractor.get_instance()


def generatePenultimateLayer(inputPathName):
    videos = []
    featureVectors = []
    for fileName in os.listdir(inputPathName):
        videos.append(os.path.join(inputPathName, fileName))
    for video in videos:
        frame = frameExtractor(video)
        feature = model.extract_feature(frame)
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
        gesture_num = lst.index(min(lst)) + 1
    return gesture_num


res = []
for x in testVectors:
    res.append(getGesture(x, trainVectors))
res = [x - 1 for x in res]
print(res)
np.savetxt('Results.csv', res, fmt="% d")
