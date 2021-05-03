# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
# code to get the key frame from the video and save it as a png file.

import cv2

# videopath : path of the video file
# frames_path: path of the directory to which the frames are saved
# count: to assign the video order to the frane.

# trainFrame = {
#     "H-0.mp4": 0.85,
#     "H-1.mp4": 0.45,
#     "H-2.mp4": 0.45,
#     "H-3.mp4": 0.5,
#     "H-4.mp4": 0.475,
#     "H-5.mp4": 0.45,
#     "H-6.mp4": 0.5,
#     "H-7.mp4": 0.5,
#     "H-8.mp4": 0.5,
#     "H-9.mp4": 0.5,
#     "H-DecreaseFanSpeed.mp4": 0.65,
#     "H-Fan_On.mp4": 0.7,
#     "H-Fan_Off.mp4": 0.6,
#     "H-IncreaseFanSpeed.mp4": 0.4,
#     "H-LightOff.mp4": 0.5,
#     "H-LightOn.mp4": 0.4,
#     "H-SetThermo.mp4": 0.6
# }


def frameExtractorSOT(videopath):
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    # video_file_name = videopath.rsplit('\\', 1)
    # cutLocation = trainFrame.get(video_file_name[1], 0.5)
    # frame_no = int(video_length * cutLocation)
    # print("Extracting frame..\n")
    frame_no = int(video_length * 0.4)
    cap.set(1, frame_no)
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

